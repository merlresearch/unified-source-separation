#!/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2024 Kohei Saijo
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0


output_dir=./fma
mkdir -p "${output_dir}"

echo "=== Preparing FMA data ==="

#################################
# Download data
#################################
# Refer to https://github.com/mdeff/fma
echo "Download FMA data"

if [ ! -e "${output_dir}/fma_medium" ]; then

    if [ ! -f "${output_dir}/fma_medium.zip" ]; then
	wget -c -P ${output_dir}  https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
	wget -c -P ${output_dir} https://os.unil.cloud.switch.ch/fma/fma_medium.zip
	echo "c67b69ea232021025fca9231fc1c7c1a063ab50b  ${output_dir}/fma_medium.zip"   | sha1sum -c -
    fi

    7z x "${output_dir}/fma_metadata.zip" -o"${output_dir}"
    7z x "${output_dir}/fma_medium.zip" -o"${output_dir}"
    # rm "${output_dir}/fma_medium.zip" "${output_dir}/fma_metadata.zip"
fi

#################################
# Metadata preparation
#################################
echo "Metadata preparation for FMA"

filename="fma_44.1k.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    python dataprep/fma.py \
	   --data_dir ${output_dir}/fma_medium \
	   --metadata_dir ${output_dir}/fma_metadata
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi
