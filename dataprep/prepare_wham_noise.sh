#!/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

output_dir=./wham_noise_48k
mkdir -p "${output_dir}"

echo "=== Preparing WHAM! noise data ==="

#################################
# WHAM! noise (48 kHz, unsegmented)
#################################
if [ ! -e "${output_dir}/high_res_wham" ]; then
    if [ ! -f "${output_dir}/high_res_wham.zip" ]; then
	echo "[WHAM! noise] downloading from https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/high_res_wham.zip"
	wget -c "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/high_res_wham.zip" -O "${output_dir}/high_res_wham.zip"
    fi

    unzip "${output_dir}/high_res_wham.zip" -d "${output_dir}"
fi

#################################
# Metadata preparation
#################################
echo "Metadata preparation for WHAM! noise"

filename="wham_noise_48k.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    python dataprep/wham_noise.py \
	   --noise_dir "${output_dir}/high_res_wham"
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi
