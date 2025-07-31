# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


def source_activity_detection(audio, L, num_chunks=10, min_power=1e-5, min_thres=1e-3):
    """Source Activity Detection (SAD) for removing silent segment.
    This follows BSRNN [1] (see Section IV-A)


    References:
        [1] Y. Luo, et al., "Music Source Separation with Band-split RNN"
            https://arxiv.org/pdf/2209.15174.pdf
    """

    # 1. split the input audio into L-sample segments with 50% overlap
    hop_size = L // 2
    segments = []
    for start in range(0, len(audio) - L + 1, hop_size):
        segments.append(audio[start : start + L])

    # 2. Split each segment into num_chunks chunks and save powers
    power_list = []
    num_zeros = 0
    for segment in segments:
        chunk_size = len(segment) // num_chunks
        chunk_powers = []
        for i in range(num_chunks):
            chunk = segment[i * chunk_size : (i + 1) * chunk_size]
            power = np.sum(chunk**2) / len(chunk)
            if power == 0:
                num_zeros += 1
                power = min_power
            chunk_powers.append(power)
        power_list.append(chunk_powers)

    # 3. 15% quantile power is used as threshold in 4.
    all_powers = np.array([p for sublist in power_list for p in sublist])
    threshold = np.percentile(all_powers, 15)
    if threshold < min_thres:
        threshold = min_thres

    # 4. Keep segments which have more than 50% chunks
    # whose powers are more than threshold
    processed_segments = []
    for seg_idx, chunk_powers in enumerate(power_list):
        if sum(p > threshold for p in chunk_powers) >= (num_chunks // 2):
            processed_segments.append(segments[seg_idx])

    return processed_segments, len(segments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--segment_length", type=int, required=True)
    parser.add_argument("--min_power", type=float, required=True)
    parser.add_argument("--min_thres", type=float, required=True)

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    # load the json file
    with open(args.json_path, "r") as f:
        metadata = json.load(f)

    # we do SAD only on training data
    train_metadata = metadata["train"]

    num_orig_segments, num_new_segments = 0, 0

    new_metadata = {}
    for key, data in tqdm(train_metadata.items()):
        audio_paths = data["ref"]
        for class_name, path in audio_paths.items():
            class_name = class_name.split("_")[0]  # e.g., drums_1 -> drums
            path = Path(path)

            if class_name not in new_metadata:
                new_metadata[class_name] = []

            audio, fs = sf.read(path, dtype="float32")
            assert fs == data["sample_rate"]

            audio = audio / np.max(np.abs(audio))

            # sad preprocessing
            processed, n_orig_seg = source_activity_detection(
                audio,
                args.segment_length * fs,
                num_chunks=args.segment_length * 2,
                min_power=args.min_power,
                min_thres=args.min_thres,
            )
            num_orig_segments += n_orig_seg
            num_new_segments += len(processed)

            # save audio files
            for i, segment in enumerate(processed):
                # e.g., drum1.wav
                filename = f"{path.stem}_{i}.wav"
                # e.g., /aaa/musdb18/train/A Classic Education - NightOwl/drums1.wav
                output_path = args.output_dir / "data" / path.parent.name
                output_path.mkdir(exist_ok=True, parents=True)
                output_path = (output_path / filename).resolve()
                sf.write(output_path, segment, fs)

                new_data = dict(
                    path=str(output_path),
                    sample_rate=fs,
                    num_samples=segment.shape[0],
                    dataset_name=data["dataset_name"],
                )
                new_metadata[class_name].append(new_data)

    metadata["train"] = new_metadata

    output_file_name = f"{args.json_path.stem}_sad.json"
    with open(args.json_path.parent / output_file_name, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Number of original segments: {num_orig_segments}")
    print(f"Number of new segments: {num_new_segments}")
