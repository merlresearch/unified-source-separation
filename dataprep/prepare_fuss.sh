#!/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

output_dir=./fuss
mkdir -p "${output_dir}"

echo "=== Preparing FUSS data ==="

#################################
# Download data
#################################
echo "Download FUSS data"

if [ ! -e "${output_dir}/ssdata" ]; then
    if [ ! -f "${output_dir}/FUSS_ssdata.tar.gz" ]; then
	wget -c -O "${output_dir}/FUSS_ssdata.tar.gz" "https://zenodo.org/records/3743844/files/FUSS_ssdata.tar.gz?download=1"
    fi

    tar xvzf "${output_dir}/FUSS_ssdata.tar.gz" -C "${output_dir}"
fi


#################################
# Metadata preparation
#################################
echo "Metadata preparation for FUSS"

filename="fuss_16k.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    python dataprep/fuss.py "${output_dir}/ssdata"
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi
