#!/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

output_dir=./musdb18/musdbhq
mkdir -p "${output_dir}"

echo "=== Preparing MUSDB-HQ data ==="

#################################
# Download data
#################################
if [[ ! -e "${output_dir}/train" || ! -e "${output_dir}/test" ]]; then
    echo "Download MUSDB-HQ data"

    if [ ! -f "${output_dir}/musdb18hq.zip" ]; then
	wget -c -O "${output_dir}/musdb18hq.zip" "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
    fi

    unzip "${output_dir}/musdb18hq.zip" -d "${output_dir}"
fi


#################################
# Metadata preparation
#################################
echo "Metadata preparation for MUSDB-HQ"

filename="musdb18hq_44.1k.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    python dataprep/musdb18hq.py "${output_dir}"
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi


filename="musdb18hq_44.1k_sad.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    # source activity detection to remove silent segments
    # segmented audios are saved to `./musdbhq_sad`
    mkdir -p ./musdbhq_sad
    python dataprep/source_activity_detection.py \
	   --json_path datasets/json_files/musdb18hq_44.1k.json \
	   --output_dir ./musdbhq_sad \
	   --segment_length 8 \
	   --min_power 1e-5 \
	   --min_thres 1e-3
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi
