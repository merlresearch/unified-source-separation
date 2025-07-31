#!/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

output_dir=./whamr
mkdir -p "${output_dir}"

echo "=== Preparing WHAM! data ==="

if [ ! -e "${output_dir}" ]; then
    echo "Please prepare WHAMR! dataset and place it to ${output_dir}"
fi


#################################
# Metadata preparation
#################################
echo "Metadata preparation for WHAM!"

filename="wham_max_16k.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    python dataprep/whamr.py \
	   --noisy_dir "${output_dir}" \
	   --sample_rate 16000 \
	   --min_or_max max \
	   --noisy
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi
