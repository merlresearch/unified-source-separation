# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

partitions = {"tr": "train", "cv": "valid", "tt": "test"}


def collect_metadata(data_dir: Path):
    ret = dict(train={}, valid={}, test={})
    for partition in ["tr", "cv", "tt"]:
        ddir = data_dir / partition
        pt = partitions[partition]

        for data_folder in tqdm(ddir.iterdir()):
            if not data_folder.is_dir():
                continue
            id_ = data_folder.name

            # load mixture and collect metadata
            num_samples = sf.info(data_folder / "mix.wav").frames
            fs = sf.info(data_folder / "mix.wav").samplerate
            assert num_samples == fs * 60, (fs, num_samples)

            data = dict(
                mix=str(data_folder / "mix.wav"),
                ref=dict(
                    speech_1=str(data_folder / "speech.wav"),
                    musicbg_1=str(data_folder / "music.wav"),
                    sfxbg_1=str(data_folder / "sfx.wav"),
                ),
                sample_rate=fs,
                num_samples=num_samples,
                dataset_name="dnr",
            )
            ret[pt][id_] = data
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", type=Path)

    args = parser.parse_args()

    metadata = collect_metadata(args.data_dir)

    filename = "dnr_44.1k.json"
    with open(f"./datasets/json_files/{filename}", "w") as f:
        json.dump(metadata, f, indent=4)
