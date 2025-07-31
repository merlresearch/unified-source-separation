# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2016 Michaël Defferrard
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

import argparse
import ast
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

partitions = {"training": "train", "validation": "valid", "test": "test"}


def load(filepath):
    """Copied from https://github.com/mdeff/fma/blob/master/utils.py#L183"""

    filename = os.path.basename(filepath)

    if "features" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "echonest" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "genres" in filename:
        return pd.read_csv(filepath, index_col=0)

    if "tracks" in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [
            ("track", "tags"),
            ("album", "tags"),
            ("artist", "tags"),
            ("track", "genres"),
            ("track", "genres_all"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [
            ("track", "date_created"),
            ("track", "date_recorded"),
            ("album", "date_created"),
            ("album", "date_released"),
            ("artist", "date_created"),
            ("artist", "active_year_begin"),
            ("artist", "active_year_end"),
        ]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ("small", "medium", "large")
        try:
            tracks["set", "subset"] = tracks["set", "subset"].astype("category", categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks["set", "subset"] = tracks["set", "subset"].astype(
                pd.CategoricalDtype(categories=SUBSETS, ordered=True)
            )

        COLUMNS = [
            ("track", "genre_top"),
            ("track", "license"),
            ("album", "type"),
            ("album", "information"),
            ("artist", "bio"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype("category")

        return tracks


@contextlib.contextmanager
def capture_stderr():
    new_err = io.StringIO()
    old_err = sys.stderr
    sys.stderr = new_err
    try:
        yield sys.stderr
    finally:
        sys.stderr = old_err


def collect_metadata(data_dir: Path, metadata_dir: Path):
    ret = dict(train=dict(musicbg=[]), valid=dict(musicbg=[]), test=dict(musicbg=[]))
    metadata_path = metadata_dir / "tracks.csv"
    tracks = load(metadata_path)
    tracks = tracks[tracks[("set", "subset")] != "large"]  # medium partition

    for split in partitions:
        tracks_split = tracks[tracks[("set", "split")] == split]

        tracks_split.reset_index(inplace=True)
        track_ids = tracks_split["track_id"]

        for track_id in tqdm(track_ids):
            # make trackid six digits
            track_id = str(track_id).zfill(6)
            audio_path = data_dir / track_id[:3] / f"{track_id}.mp3"

            with capture_stderr() as stderr:
                try:
                    # some mp3 files is broken for some reason
                    # which can sometimes be detected only by reading the file
                    audio, fs = sf.read(audio_path)
                    num_samples = audio.shape[0]

                    if abs(audio).sum() == 0.0:
                        print(f"Zero-signal {audio_path}")
                        continue

                except sf.LibsndfileError:
                    print(f"Error reading {audio_path}")
                    continue

                # some mp3 files is broken for some reason
                # which can sometimes be detected only by reading the file
                stderr_output = stderr.getvalue()
                if "dequantization failed" in stderr_output:
                    print("Specific issue detected: dequantization failed.")
                    continue
                elif "too large for available bit count" in stderr_output:
                    print("Specific issue detected:")
                    continue

            data = dict(
                path=str(audio_path),
                sample_rate=fs,
                num_samples=num_samples,
                dataset_name="fma",
            )

            ret[partitions[split]]["musicbg"].append(data)

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--metadata_dir", type=Path, required=True)

    args = parser.parse_args()

    metadata = collect_metadata(args.data_dir, args.metadata_dir)
    filename = "fma_44.1k.json"
    with open(f"./datasets/json_files/{filename}", "w") as f:
        json.dump(metadata, f, indent=4)
