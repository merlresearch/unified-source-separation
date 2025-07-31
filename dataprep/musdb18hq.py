# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

dev_names = [
    "Actions - One Minute Smile",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Johnny Lokke - Promises & Lies",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
    "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants",
    "Leaf - Summerghost",
    "Skelpolu - Human Mistakes",
    "Young Griffo - Pennies",
    "ANiMAL - Rockshow",
    "James May - On The Line",
    "Meaxic - Take A Step",
    "Traffic Experiment - Sirens",
]


def collect_metadata(data_dir: Path):
    ret = dict(train={}, valid={}, test={})
    for partition in ["train", "valid", "test"]:
        ddir = data_dir / "test" if partition == "test" else data_dir / "train"

        for data_folder in tqdm(ddir.iterdir()):
            if not data_folder.is_dir():
                continue
            id_ = data_folder.name

            if partition == "train" and id_ in dev_names:
                continue
            elif partition == "valid" and id_ not in dev_names:
                continue

            # load mixture and collect metadata
            num_samples = sf.info(data_folder / "mixture.wav").frames
            fs = sf.info(data_folder / "mixture.wav").samplerate

            data = dict(
                mix=str(data_folder / "mixture.wav"),
                ref=dict(
                    drums_1=str(data_folder / "drums.wav"),
                    bass_1=str(data_folder / "bass.wav"),
                    vocals_1=str(data_folder / "vocals.wav"),
                    other_1=str(data_folder / "other.wav"),
                ),
                sample_rate=fs,
                num_samples=num_samples,
                dataset_name="musdb18hq",
            )
            ret[partition][id_] = data
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # /db/original/public/musdb18/musdbhq
    parser.add_argument("data_dir", type=Path)

    args = parser.parse_args()

    metadata = collect_metadata(args.data_dir)

    filename = "musdb18hq_44.1k.json"
    with open(f"./datasets/json_files/{filename}", "w") as f:
        json.dump(metadata, f, indent=4)
