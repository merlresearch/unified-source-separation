#!/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


output_dir=./vctk_demand
mkdir -p "${output_dir}"

echo "=== Preparing VCTK-DEMAND data ==="

subsets=(
    "clean_trainset_28spk_wav"
    "noisy_trainset_28spk_wav"
    "clean_testset_wav"
    "noisy_testset_wav"
)
sequence_nums=(2 6 1 5)

#################################
# Download data
#################################

length=${#subsets[@]}

for ((i=0; i<$length; i++)); do
    subset=${subsets[$i]}
    sequence=${sequence_nums[$i]}

    if [ ! -e "${output_dir}/${subset}" ]; then
	if [ ! -f "${output_dir}/${subset}.zip" ]; then
            wget -c -O "${output_dir}/${subset}.zip" "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/${subset}.zip?sequence=${sequence}&isAllowed=y"
            # axel -a -n 10 -o "${output_dir}/${subset}" "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/${subset}?sequence=${sequence}&isAllowed=y"
	fi

        unzip "${output_dir}/${subset}.zip" -d "${output_dir}"
    fi
done

#################################
# Metadata preparation
#################################
echo "Metadata preparation for VCTK-DEMAND"

filename="vctk_demand_48k.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    python dataprep/vctk_demand.py \
	   --noisy_dir ${output_dir}
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi
