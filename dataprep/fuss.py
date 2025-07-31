# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

partitions = {"train": "train", "validation": "valid", "eval": "test"}
num_data = {"train": 20000, "validation": 1000, "eval": 1000}


def collect_metadata(data_dir: Path):
    ret = dict(train={}, valid={}, test={})
    for partition in partitions:
        ddir = data_dir / partition
        pt = partitions[partition]

        for idx in tqdm(range(num_data[partition])):
            id_ = f"example{idx:05}" if partition == "train" else f"example{idx:04}"

            # load mixture and collect metadata
            mix_path = ddir / f"{id_}.wav"
            num_samples = sf.info(mix_path).frames
            fs = sf.info(mix_path).samplerate

            # some sources in fuss is zero signals
            # they are filtered out here
            valid_data = []
            for wav_path in (ddir / f"{id_}_sources").iterdir():
                audio, fs2 = sf.read(wav_path)
                assert fs == fs2, (fs, fs2)

                if abs(audio).sum() == 0.0:
                    print(f"zero signal: {str(wav_path)}")
                    continue

                valid_data.append(wav_path)

            if not valid_data:
                print(f"No audios were active: {id_}")
                continue

            ref = {}
            for i, path in enumerate(valid_data):
                ref[f"sfx_{i+1}"] = str(path)

            data = dict(
                mix=str(mix_path),
                ref=ref,
                sample_rate=fs,
                num_samples=num_samples,
                dataset_name="fuss",
            )
            ret[pt][id_] = data
    return ret


def split_with_num_src(metadata):
    ret = [dict(train={}, valid={}, test={}) for _ in range(4)]

    for partition in partitions.values():
        for id_, data in metadata[partition].items():
            n_src = len(data["ref"])
            ret[n_src - 1][partition][id_] = data

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", type=Path)

    args = parser.parse_args()

    metadata = collect_metadata(args.data_dir)
    filename = "fuss_16k.json"
    with open(f"./datasets/json_files/{filename}", "w") as f:
        json.dump(metadata, f, indent=4)

    with open("datasets/json_files/fuss_16k.json", "r") as f:
        metadata = json.load(f)
    metadata = split_with_num_src(metadata)
    for n in range(4):
        filename = f"fuss_{n+1}src_16k.json"
        with open(f"./datasets/json_files/{filename}", "w") as f:
            json.dump(metadata[n], f, indent=4)
