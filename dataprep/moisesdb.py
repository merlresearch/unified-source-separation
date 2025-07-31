# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def collect_metadata(data_dir, split):
    ret = dict(train={}, valid={}, test={})
    for partition in ["train", "valid", "test"]:
        split_part = split[partition]

        for id_ in tqdm(split_part):
            this_data_dir = data_dir / id_

            if not this_data_dir.exists():
                raise FileNotFoundError(f"{this_data_dir} does not exist")
            if not this_data_dir.is_dir():
                raise RuntimeError(f"{this_data_dir} is not a directory. Something's wrong.")

            with open(this_data_dir / "data.json", "r") as f:
                metadata = json.load(f)["stems"]

            data_tmp = dict(bass=[], drums=[], other=[], vocals=[])
            for each_stem in metadata:
                stem_name = each_stem["stemName"]
                for track in each_stem["tracks"]:
                    stem_id = track["id"]
                    path = this_data_dir / stem_name / f"{stem_id}.wav"

                    if stem_name in data_tmp:
                        data_tmp[stem_name].append(str(path))
                    else:
                        data_tmp["other"].append(str(path))

            # load mixture and collect
            num_samples = sf.info(path).frames
            fs = sf.info(path).samplerate

            data = dict(ref={}, sample_rate=fs, num_samples=num_samples, dataset_name="moisesdb")
            for stem, paths in data_tmp.items():
                for i, path in enumerate(paths):
                    data["ref"][f"{stem}_{i+1}"] = path

            ret[partition][id_] = data
    return ret


def split_from_csv(csv_path):
    """
    Return a dictionary with the split assignment for each track id.
    """
    assignment = {1: "train", 2: "train", 3: "train", 4: "valid", 5: "test"}
    ret = dict(train=[], valid=[], test=[])
    with open(csv_path, "r") as f:
        for i, line in enumerate(f):
            # skip the header
            if i == 0:
                continue

            # line[0] is the track id, line[1] is the split number
            line = line.strip().split(",")
            split = assignment[int(line[1])]

            # append the track id to the corresponding split
            ret[split].append(line[0])
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # /db/original/public/moisesdb/moisesdb_v0.1
    parser.add_argument("data_dir", type=Path)
    # https://github.com/kwatcharasupat/query-bandit/blob/main/reproducibility/splits.csv
    # ./datasets/dataset_file_makers/moisesdb_split.csv
    parser.add_argument("split_csv", type=Path)

    args = parser.parse_args()

    split = split_from_csv(args.split_csv)
    metadata = collect_metadata(args.data_dir, split=split)

    filename = "moisesdb_44.1k.json"
    with open(f"./datasets/json_files/{filename}", "w") as f:
        json.dump(metadata, f, indent=4)
