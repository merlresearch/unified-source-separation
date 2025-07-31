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
dev_spks = ["p226", "p287"]
test_spks = ["p232", "p257"]


def collect_speech_metadata(speech_dir: Path):
    # collect speech metadata
    ret = []
    for spk_dir in tqdm(speech_dir.iterdir()):
        # skip speakers in dev and test
        if spk_dir.name in dev_spks or spk_dir.name in test_spks:
            continue

        for wav_path in spk_dir.glob("*_mic1.flac"):
            num_samples = sf.info(wav_path).frames
            fs = sf.info(wav_path).samplerate

            data = dict(
                path=str(wav_path),
                sample_rate=fs,
                num_samples=num_samples,
                dataset_name="vctk_speech",
            )
            ret.append(data)
    return ret


def collect_metadata(speech_dir: Path):
    ret = dict(train={}, valid={}, test={})
    ret["train"]["speech"] = collect_speech_metadata(speech_dir)

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # /db/original/public/VCTK-Corpus/wav48
    parser.add_argument("--speech_dir", type=Path, required=True)

    args = parser.parse_args()

    metadata = collect_metadata(args.speech_dir)

    filename = "vctk_48k.json"
    with open(f"./datasets/json_files/{filename}", "w") as f:
        json.dump(metadata, f, indent=4)
