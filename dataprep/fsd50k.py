# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm


def extract_data_with_one_label(ground_truth_csv, collection_csv):
    df_gt = pd.read_csv(ground_truth_csv)
    df_cl = pd.read_csv(collection_csv)
    df_cl["label_count"] = df_cl["labels"].apply(lambda x: 1 if isinstance(x, float) or len(x.split(",")) == 1 else 0)

    df_one_label = df_gt[df_cl["label_count"] == 1]
    df_multi_label = df_gt[df_cl["label_count"] != 1]
    print(f"Number of samples with one label: {len(df_one_label)}")
    print(f"Number of samples with multiple labels: {len(df_multi_label)}")
    return df_one_label, df_multi_label


def remove_music_and_speech(df_one_label, ontology_path):
    # https://research.google.com/audioset/ontology/index.html

    with open(ontology_path, "r") as f:
        ontology = json.load(f)
    ontology = {d["id"]: d for d in ontology}

    speech_ancestors = [
        "/m/09l8g",
        "/m/09hlz4",
        "/t/dd00012",
    ]  # Human Voice, Respiratory sounds, Human group actions
    music_ancestors = ["/m/04rlf"]  # Music

    def collect_all_child_ids(ids, ontology):
        result = []

        def recursive_collect(current_id):
            if current_id in ontology:
                children = ontology[current_id]["child_ids"]
                result.extend(children)
                for child in children:
                    recursive_collect(child)

        for id in ids:
            recursive_collect(id)
        return result

    speech_children = collect_all_child_ids(speech_ancestors, ontology)
    music_children = collect_all_child_ids(music_ancestors, ontology)
    music_and_speech = speech_children + music_children

    def filter_with_id(row):
        mids = row["mids"].split(",")
        for mid in mids:
            if mid in music_and_speech:
                return True
        return False

    df_one_label = df_one_label[~df_one_label.apply(filter_with_id, axis=1)]
    print(f"Number of samples without music and speech: {len(df_one_label)}")
    return df_one_label


def split_train_val(df_one_label):
    df_train = df_one_label[df_one_label["split"] == "train"]
    df_valid = df_one_label[df_one_label["split"] == "val"]
    return df_train, df_valid


def collect_metadata(audio_root_dir, df, fore_back_border=8.0):
    # fore_back_border:
    #    The border between foreground and background in seconds.
    #    Audios shorter than this value are considered as foreground.

    ret_fore, ret_back = [], []
    for index, row in tqdm(df.iterrows()):
        audio_path = audio_root_dir / f"{row['fname']}.wav"
        num_samples = sf.info(audio_path).frames
        fs = sf.info(audio_path).samplerate

        data = dict(
            path=str(audio_path),
            sample_rate=fs,
            num_samples=num_samples,
            labels=row["labels"],
            mids=row["mids"],
            dataset_name="fsd50k",
        )

        duration = num_samples / fs
        if duration < fore_back_border:
            ret_fore.append(data)
        else:
            ret_back.append(data)

    return ret_fore, ret_back


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--ontology_path", type=Path, required=True)

    args = parser.parse_args()

    metadata = dict(train={}, valid={}, test={})
    for split in ["dev"]:
        print(f"=== Process {split} set ===")

        df_one_src, df_multi_src = extract_data_with_one_label(
            args.data_dir / "FSD50K.ground_truth" / f"{split}.csv",
            args.data_dir / "FSD50K.metadata" / "collection" / f"collection_{split}.csv",
        )

        # remove speech and music
        df_one_src = remove_music_and_speech(df_one_src, args.ontology_path)
        df_multi_src = remove_music_and_speech(df_multi_src, args.ontology_path)

        if split == "dev":
            # process single-source audio first
            df_one_src_train, df_one_src_valid = split_train_val(df_one_src)

            # train set
            data_one_src_train_fore, data_one_src_train_back = collect_metadata(
                args.data_dir / "FSD50K.dev_audio", df_one_src_train
            )
            metadata["train"]["sfx"] = data_one_src_train_fore
            metadata["train"]["sfxbg"] = data_one_src_train_back

            # validation set
            data_one_src_valid_fore, data_one_src_valid_back = collect_metadata(
                args.data_dir / "FSD50K.dev_audio", df_one_src_valid
            )
            metadata["valid"]["sfx"] = data_one_src_valid_fore
            metadata["valid"]["sfxbg"] = data_one_src_valid_back

            # process multi-source audio
            # multi-source foregrounds are discarded
            df_multi_src_train, df_multi_src_valid = split_train_val(df_multi_src)

            # train set
            data_multi_src_train_fore, data_multi_src_train_back = collect_metadata(
                args.data_dir / "FSD50K.dev_audio", df_multi_src_train
            )
            metadata["train"]["sfxbg"] += data_multi_src_train_back

            # validation set
            data_multi_src_valid_fore, data_multi_src_valid_back = collect_metadata(
                args.data_dir / "FSD50K.dev_audio", df_multi_src_valid
            )
            metadata["valid"]["sfxbg"] += data_multi_src_valid_back

        else:
            # single-source audio
            data_one_src_test_fore, data_one_src_test_back = collect_metadata(
                args.data_dir / "FSD50K.eval_audio", df_one_src
            )
            metadata["test"]["sfx"] = data_one_src_test_fore
            metadata["test"]["sfxbg"] = data_one_src_test_back

            # multi-source audio
            data_multi_src_test_fore, data_multi_src_test_back = collect_metadata(
                args.data_dir / "FSD50K.eval_audio", df_multi_src
            )
            metadata["test"]["sfxbg"] += data_multi_src_test_back

    for split, data in metadata.items():
        for prompt, d in data.items():
            print(f"{split} {prompt}: {len(d)}")

    filename = "fsd50k_44.1k.json"
    with open(f"./datasets/json_files/{filename}", "w") as f:
        json.dump(metadata, f, indent=4)
