#!/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

output_dir=./vctk
mkdir -p "${output_dir}"

echo "=== Preparing VCTK data ==="
#################################
# Download data
#################################
# Refer to https://datashare.ed.ac.uk/handle/10283/3443
if [ ! -d "${output_dir}/VCTK-Corpus" ]; then
    echo "Downloading VCTK data"

    if [ ! -f "${output_dir}/DS_10283_3443.zip" ]; then
	wget --continue "https://datashare.ed.ac.uk/download/DS_10283_3443.zip" -O "${output_dir}/DS_10283_3443.zip"
    fi

    if [ ! -f "${output_dir}/VCTK-Corpus-0.92.zip" ]; then
	echo "Unzip DS_10283_3443.zip file"
	UNZIP_DISABLE_ZIPBOMB_DETECTION=1 unzip "${output_dir}/DS_10283_3443.zip" -d "${output_dir}"
    else
	echo "Skip unzipping DS_10283_3443.zip file"
    fi

    echo "Unzip VCTK-Corpus-0.92.zip file"
    UNZIP_DISABLE_ZIPBOMB_DETECTION=1 unzip "${output_dir}/VCTK-Corpus-0.92.zip" -d "${output_dir}/VCTK-Corpus"
else
    echo "${output_dir}/VCTK-Corpus already exists, skipping download"
fi

#################################
# Metadata preparation
#################################
echo "Metadata preparation for VCTK"

filename="vctk_48k.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    python dataprep/vctk.py \
	   --speech_dir "${output_dir}/VCTK-Corpus/wav48_silence_trimmed"
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi
