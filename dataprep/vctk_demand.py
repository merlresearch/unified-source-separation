# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

dev_spks = ["p226", "p287"]


def collect_noisy_metadata(root_dir: Path, stage: str):
    if stage == "valid":
        mix_dir = root_dir / "noisy_trainset_28spk_wav"
        speech_dir = root_dir / "clean_trainset_28spk_wav"
        noise_dir = root_dir / "noise_trainset_28spk_wav"
    else:
        mix_dir = root_dir / "noisy_testset_wav"
        speech_dir = root_dir / "clean_testset_wav"
        noise_dir = root_dir / "noise_testset_wav"

    noise_dir.mkdir(exist_ok=True, parents=True)

    # collect noisy metadata
    ret = {}
    for wav_path in tqdm(mix_dir.glob("*.wav")):
        if stage == "valid" and wav_path.stem[:4] not in dev_spks:
            continue

        mix, fs = sf.read(wav_path)
        num_samples = mix.shape[0]

        speech_path = speech_dir / wav_path.name
        speech, _ = sf.read(speech_path)

        # noise is not provided, so get it by subtracting speech from mix
        noise = mix - speech
        noise_path = noise_dir / wav_path.name
        sf.write(noise_path, noise, fs)

        data = dict(
            mix=str(wav_path),
            ref=dict(speech_1=str(speech_path), sfxbg_1=str(noise_path)),
            sample_rate=fs,
            num_samples=num_samples,
            dataset_name="vctk_demand",
        )
        ret[wav_path.name] = data
    return ret


def collect_metadata(noisy_dir: Path):
    ret = dict(train={}, valid={}, test={})
    ret["valid"] = collect_noisy_metadata(noisy_dir, "valid")
    ret["test"] = collect_noisy_metadata(noisy_dir, "test")

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--noisy_dir", type=Path, required=True)

    args = parser.parse_args()

    metadata = collect_metadata(args.noisy_dir)

    filename = "vctk_demand_48k.json"
    with open(f"./datasets/json_files/{filename}", "w") as f:
        json.dump(metadata, f, indent=4)
