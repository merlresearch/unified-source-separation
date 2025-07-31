# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from nets.model_wrapper import SeparationModel
from utils.audio_utils import resample
from utils.average_model_params import average_model_params
from utils.config import yaml_to_parser

RESAMPLE_RATE = 48000

# parameters used to plot the spectrogram
n_fft = 512
hop_length = 128


def plot_fig(data, save_path, title):
    if isinstance(data, torch.Tensor):
        data = data.clone().cpu().numpy()

    # plot spectrogram
    fig, ax = plt.subplots(figsize=(6, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    img = librosa.display.specshow(
        D,
        y_axis="linear",
        x_axis="time",
        sr=fs,
        ax=ax,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    ax.set(title=title)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Frequency [hz]")
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.subplots_adjust(left=0.125, right=0.95, top=0.88, bottom=0.11)
    plt.savefig(save_path)
    plt.close()


def separate(args):

    allowed_suffix = [".ckpt", ".pth"]
    if args.ckpt_path.is_dir():
        assert (args.ckpt_path / "checkpoints").exists(), f"{args.ckpt_path} must contain checkpoints directory."
        config_path = args.ckpt_path / "hparams.yaml"
        ckpt_paths = [
            path
            for path in Path(args.ckpt_path / "checkpoints").iterdir()
            if path.suffix in allowed_suffix and path.name != "last.ckpt"
        ]
        assert all(
            [ckpt_paths[0].suffix == p.suffix for p in ckpt_paths]
        ), "all the suffix of the pre-trained weights files must be the same"
    else:
        assert args.ckpt_path.suffix in allowed_suffix, f"Only {allowed_suffix} files are supported."
        config_path = args.ckpt_path.parent.parent / "hparams.yaml"
        ckpt_paths = [args.ckpt_path]

    # instantiate the model
    hparams = yaml_to_parser(config_path)
    hparams = hparams.parse_args([])
    model = SeparationModel(
        hparams.encoder_name,
        hparams.encoder_conf,
        hparams.decoder_name,
        hparams.decoder_conf,
        hparams.model_name,
        hparams.model_conf,
        hparams.css_conf,
        hparams.variance_normalization,
    )

    state_dict = average_model_params(ckpt_paths)
    new_state_dict = {}
    for key, value in state_dict.items():
        k = key.replace("model.", "")
        new_state_dict[k] = value
    model.load_state_dict(new_state_dict)
    model.cuda()

    mix, fs = torchaudio.load(args.audio_path)

    start_sample = max(0, args.start_time) * fs
    if args.end_time < 0:  # process the whole signal
        end_sample = -1
    else:
        end_time = max(args.start_time + 1, args.end_time)
        end_sample = end_time * fs

    mix = mix[[args.ref_channel], start_sample:end_sample]
    mix_return = mix.clone()
    mix = mix.cuda()

    # in some models, we want to always up-/down-sample the input
    if RESAMPLE_RATE != fs:
        mix = resample(mix, fs, RESAMPLE_RATE)

    # separation
    print(f"Separate these sources: {args.prompts}")
    with torch.no_grad():
        # Chunk-wise processing when args.css_segment_size is given
        # NOTE: model.css does not solve inter-chunk permutation
        if args.css_segment_size is not None:
            model.css_segment_size = args.css_segment_size
            if args.css_shift_size is None:
                model.css_shift_size = args.css_segment_size // 2
            else:
                model.css_shift_size = args.css_shift_size

            y, *_ = model.css(mix, [args.prompts])  # (batch, n_samples) -> (n_src, n_samples)
        else:
            y, *_ = model(mix, [args.prompts])  # (batch, n_samples) -> (n_src, n_samples)

    # back to original sampling rate
    if RESAMPLE_RATE != fs:
        y = resample(y, RESAMPLE_RATE, fs)

    return y.cpu(), mix_return, fs


if __name__ == "__main__":
    """Basic usage
    python separate.py \
        --ckpt_path /path/to/model.pth \
        --audio_path /path/to/audio.wav \
        --prompts speech sfx sfx musicbg \
        --audio_output_dir /path/to/output/dir
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_path",
        type=Path,
        required=True,
        help="Path to pre-trained model parameters (.pth file).",
    )
    parser.add_argument(
        "--audio_path",
        type=Path,
        required=True,
        help="Path to the audio file to separate.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        required=True,
        help="List of prompts indicating the sources to be separated.",
    )
    parser.add_argument(
        "--audio_output_dir",
        type=Path,
        default="./audio_outputs",
        help="Directory to save the separated audios.",
    )
    parser.add_argument(
        "--css_segment_size",
        type=int,
        required=False,
        default=None,
        help="When given, chunk-wise processing (CSS) is done instead of full-utterance processing."
        "Normal full-utterance processing is done when this is set to None, even if css_shift_size is given.",
    )
    parser.add_argument(
        "--css_shift_size",
        type=int,
        required=False,
        default=None,
        help="Shift size of CSS. When not given, shift size is set to half of the css_segment_size.",
    )
    parser.add_argument(
        "--start_time",
        type=int,
        required=False,
        default=0,
        help="Start time of the segment to process in seconds.",
    )
    parser.add_argument(
        "--end_time",
        type=int,
        required=False,
        default=-1,
        help="End time of the segment to process in seconds. Set to negative value to process whole signal",
    )
    parser.add_argument(
        "--ref_channel",
        type=int,
        required=False,
        default=0,
        help="Reference channel for multichannel input.",
    )

    args = parser.parse_args()

    args.audio_output_dir.mkdir(exist_ok=True, parents=True)

    est, mix, fs = separate(args)

    scale = torch.max(abs(mix)) / 0.95
    mix /= scale
    est /= scale

    audio_output_dir = args.audio_output_dir / "wav"
    spec_output_dir = args.audio_output_dir / "spectrogram"

    audio_output_dir.mkdir(exist_ok=True, parents=True)
    spec_output_dir.mkdir(exist_ok=True, parents=True)

    torchaudio.save(audio_output_dir / "mix.wav", mix, fs)
    plot_fig(mix[0], spec_output_dir / "mix.png", "Input mixture")

    for i, (e, p) in enumerate(zip(est, args.prompts)):
        torchaudio.save(audio_output_dir / f"{p}{i}.wav", e[None], fs)
        plot_fig(e, spec_output_dir / f"{p}{i}.png", f"{p}")
