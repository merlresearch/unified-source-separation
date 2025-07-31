# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import json
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def collect_speech_metadata(speech_dir: Path, speech_metadata_dir: Path):
    assignment = {}
    for subset in speech_dir.iterdir():
        if not (subset / "wsj0" / "si_tr_s").exists():
            continue
        for spk_subset in (subset / "wsj0" / "si_tr_s").iterdir():
            assignment[spk_subset.name] = subset.name

    speech_paths = []
    txt_file = speech_metadata_dir / "mix_2_spk_tr.txt"
    with open(txt_file, "r") as f:
        for line in f.readlines():
            speech_paths.append(line.split(" ")[0])
            speech_paths.append(line.split(" ")[2])

    # remove duplicated paths
    speech_paths = list(set(speech_paths))

    # collect speech metadata
    ret = []
    for wav_path in tqdm(speech_paths):
        spk_id = wav_path.split("/")[2]
        wav_path = speech_dir / assignment[spk_id] / wav_path

        num_samples = sf.info(wav_path).frames
        fs = sf.info(wav_path).samplerate

        data = dict(
            path=str(wav_path),
            sample_rate=fs,
            num_samples=num_samples,
            dataset_name="wsj0",
        )
        ret.append(data)
    return ret


def collect_metadata(
    speech_dir: Path,
    speech_metadata_dir: Path,
):
    ret_dm = dict(train={}, valid={}, test={})  # for dynamic mixing setup
    ret_dm["train"]["speech"] = collect_speech_metadata(speech_dir, speech_metadata_dir)
    return ret_dm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speech_dir",
        type=Path,
        required=True,
        help="Path to WSJ0 directory",
    )
    parser.add_argument(
        "--speech_metadata_dir",
        type=Path,
        required=True,
        help="Path to WSJ0-2mix metadata directory",
    )
    args = parser.parse_args()

    metadata = collect_metadata(
        args.speech_dir,
        args.speech_metadata_dir,
    )

    with open("./datasets/json_files/wsj0_speech_16k.json", "w") as f:
        json.dump(metadata, f, indent=4)
