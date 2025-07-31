# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def collect_noisy_metadata(root_dir: Path, stage: str, noisy: bool = False, reverberant: bool = False):
    root_dir = root_dir / stage

    if noisy and reverberant:
        dset_name = "whamr"
        mix_dir = root_dir / "mix_both_reverb"
    elif noisy and not reverberant:
        dset_name = "wham"
        mix_dir = root_dir / "mix_both_anechoic"
    else:
        dset_name = "wsj0-2mix"
        mix_dir = root_dir / "mix_clean_anechoic"

    # collect noisy metadata
    ret = {}
    for wav_path in tqdm(mix_dir.glob("*.wav")):
        num_samples = sf.info(wav_path).frames
        fs = sf.info(wav_path).samplerate

        speech_path1 = root_dir / "s1_anechoic" / wav_path.name
        speech_path2 = root_dir / "s2_anechoic" / wav_path.name
        noise_path = root_dir / "noise" / wav_path.name

        data = dict(
            mix=str(wav_path),
            ref=dict(
                speech_1=str(speech_path1),
                speech_2=str(speech_path2),
            ),
            sample_rate=fs,
            num_samples=num_samples,
            dataset_name=dset_name,
        )

        if noisy:
            data["ref"]["sfxbg_1"] = str(noise_path)

        ret[wav_path.name] = data
    return ret


def collect_metadata(
    noisy_dir: Path,
    noisy: bool = False,
    reverberant=False,
):
    ret = dict(train={}, valid={}, test={})  # for static mixing setup

    tr_data = collect_noisy_metadata(noisy_dir, "tr", noisy=noisy, reverberant=reverberant)
    cv_data = collect_noisy_metadata(noisy_dir, "cv", noisy=noisy, reverberant=reverberant)
    tt_data = collect_noisy_metadata(noisy_dir, "tt", noisy=noisy, reverberant=reverberant)

    ret["train"] = tr_data
    ret["valid"] = cv_data
    ret["test"] = tt_data

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--noisy_dir", type=Path, required=True, help="Path to WHAMR! directory")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--min_or_max", type=str, default="max", choices=["min", "max"])
    parser.add_argument("--noisy", action="store_true")
    parser.add_argument("--reverberant", action="store_true")

    args = parser.parse_args()

    assert args.sample_rate == 16000, "Only 16kHz is supported"
    sf_str = f"{args.sample_rate//1000}k"

    noisy_dir = args.noisy_dir / f"wav{args.sample_rate//1000}k" / args.min_or_max

    metadata = collect_metadata(
        noisy_dir,
        args.noisy,
        args.reverberant,
    )

    if args.noisy and args.reverberant:
        dset_name = "whamr"
    elif args.noisy and not args.reverberant:
        dset_name = "wham"
    else:
        dset_name = "wsj0-2mix"

    filename = f"{dset_name}_{args.min_or_max}_{sf_str}.json"
    with open(f"./datasets/json_files/{filename}", "w") as f:
        json.dump(metadata, f, indent=4)
