# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def collect_metadata(speech_scp: Path):
    with open(speech_scp) as f:
        lines = f.readlines()

    paths = [line.split()[-1].strip() for line in lines]

    ret = []
    for path in tqdm(paths):
        num_samples = sf.info(path).frames
        fs = sf.info(path).samplerate

        data = dict(
            path=str(path),
            sample_rate=fs,
            num_samples=num_samples,
            dataset_name="librivox_urgent2024",
        )
        ret.append(data)

    return {"train": {"speech": ret}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--speech_scp", type=Path, required=True)

    args = parser.parse_args()

    metadata = collect_metadata(args.speech_scp)

    filename = "librivox_urgent2024.json"
    with open(f"./datasets/json_files/{filename}", "w") as f:
        json.dump(metadata, f, indent=4)
