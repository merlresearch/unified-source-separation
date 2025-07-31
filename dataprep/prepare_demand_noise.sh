#!/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

output_dir=./demand

noise_types=(
    DKITCHEN
    DLIVING
    DWASHING
    NFIELD
    NPARK
    NRIVER
    OHALLWAY
    OMEETING
    OOFFICE
    PCAFETER
    PRESTO
    PSTATION
    SCAFE
    SPSQUARE
    STRAFFIC
    TBUS
    TCAR
    TMETRO
)

#################################
# Download data
#################################
echo "Download DEMAND noise"

mkdir -p "${output_dir}/zip_files"
for noise_type in ${noise_types[@]}; do
    if [ ! -e "${output_dir}/${noise_type}" ]; then
	if [ ! -f "${output_dir}/zip_files/${noise_type}_48k.zip" ]; then
            wget -O "${output_dir}/zip_files/${noise_type}_48k.zip" https://zenodo.org/records/1227121/files/${noise_type}_48k.zip?download=1
	fi
        unzip "${output_dir}/zip_files/${noise_type}_48k.zip" -d "${output_dir}"
    fi
done

#################################
# Metadata preparation
#################################
echo "Metadata preparation for DEMAND"

filename="demand_48k.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    python dataprep/demand.py \
	   --noise_dir "${output_dir}"
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi
