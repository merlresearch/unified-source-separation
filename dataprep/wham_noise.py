# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def collect_noise_metadata(noise_dir: Path, seg_len=8, seg_shift=4):
    filenames = []
    with open(noise_dir / "high_res_metadata.csv", "r") as f:
        headers = f.readline().strip().split(",")
        fname_idx = headers.index("Filename")
        subset_idx = headers.index("WHAM! Split")
        for line in f:
            tup = line.strip().split(",")
            if tup[subset_idx] == "Train":
                filenames.append(Path(tup[fname_idx]).name)

    # collect noise metadata
    ret = []
    noise_dir = noise_dir / "audio"
    for filename in tqdm(filenames):
        output_dir = noise_dir.parent / "high_res_wham_segment"
        output_dir.mkdir(exist_ok=True, parents=True)

        wav_path = noise_dir / filename
        wav, fs = sf.read(wav_path)
        num_samples = len(wav)

        seg_output_dir = output_dir / f"{filename[:-4]}"
        (seg_output_dir).mkdir(exist_ok=True, parents=True)

        num_seg = (num_samples - seg_len * fs) // (seg_shift * fs) + 1
        for i in range(num_seg):
            start = i * seg_shift * fs
            end = start + seg_len * fs
            seg_wav_path = seg_output_dir / f"{i:03d}.wav"
            sf.write(seg_wav_path, wav[start:end], fs)

            data = dict(
                path=str(seg_wav_path),
                sample_rate=fs,
                num_samples=end - start,
                dataset_name="wham_noise",
            )
            ret.append(data)
    return ret


def collect_metadata(
    noise_dir: Path,
):
    ret_dm = dict(train={}, valid={}, test={})  # for dynamic mixing setup
    ret_dm["train"]["sfxbg"] = collect_noise_metadata(noise_dir)
    return ret_dm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--noise_dir",
        type=Path,
        required=True,
        help="Path to 48k unsegmented WHAM! noise",
    )
    args = parser.parse_args()

    metadata = collect_metadata(args.noise_dir)

    with open("./datasets/json_files/wham_noise_48k.json", "w") as f:
        json.dump(metadata, f, indent=4)
