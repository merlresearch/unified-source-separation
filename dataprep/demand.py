# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

# see this paper for noise partition
# https://www.isca-archive.org/interspeech_2016/valentinibotinhao16_interspeech.pdf
test_noise = ["DLIVING", "OOFFICE", "TBUS", "PCAFETER", "SPSQUARE"]


def collect_noise_metadata(noise_dir: Path, seg_len=8, seg_shift=4):
    # collect noise metadata
    ret = []

    for n_dir in tqdm(noise_dir.iterdir()):
        # skip speakers in dev and test
        if n_dir.name in test_noise:
            continue
        if n_dir.name in ["zip_files", "demand_48k_segment"]:
            continue

        output_dir = noise_dir / "demand_48k_segment"
        output_dir.mkdir(exist_ok=True, parents=True)

        wav_path = n_dir / "ch01.wav"
        wav, fs = sf.read(wav_path)
        num_samples = len(wav)

        num_seg = (num_samples - seg_len * fs) // (seg_shift * fs) + 1
        for i in range(num_seg):
            start = i * seg_shift * fs
            end = start + seg_len * fs
            seg_wav_path = output_dir / f"{n_dir.name}_{i:03d}.wav"
            sf.write(seg_wav_path, wav[start:end], fs)

            data = dict(
                path=str(seg_wav_path),
                sample_rate=fs,
                num_samples=end - start,
                dataset_name="vctk_demand",
            )
            ret.append(data)
    return ret


def collect_metadata(noise_dir: Path):
    ret = dict(train={}, valid={}, test={})
    ret["train"]["sfxbg"] = collect_noise_metadata(noise_dir)

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--noise_dir", type=Path, required=True)

    args = parser.parse_args()

    metadata = collect_metadata(args.noise_dir)

    filename = "demand_48k.json"
    with open(f"./datasets/json_files/{filename}", "w") as f:
        json.dump(metadata, f, indent=4)
