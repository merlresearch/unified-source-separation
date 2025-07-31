#!/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

output_dir=./dnr
mkdir -p "${output_dir}"

echo "=== Preparing DnR data ==="

#################################
# Download data
#################################
if [ ! -e "${output_dir}/dnr_v2" ]; then
    if [ ! -f "${output_dir}/dnr_v2.tar.gz" ]; then
	echo "Download DnR data"
        seq -w 00 10 | xargs -I {} -P 11 bash -c '
            ii="$1"
            outdir="$2"
            wget -c -O ${outdir}/dnr_v2.tar.gz.${ii} https://zenodo.org/records/6949108/files/dnr_v2.tar.gz.${ii}?download=1
	        ' _ {} "${output_dir}"

	cat ${output_dir}/dnr_v2.tar.gz.* > "${output_dir}/dnr_v2.tar.gz"
    fi

    echo "Untar the DnR data"
    tar xvzf "${output_dir}/dnr_v2.tar.gz" -C "${output_dir}"
fi
#
# #################################
# # Metadata preparation
# #################################
echo "Metadata preparation for DnR"

filename="dnr_44.1k.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    python dataprep/dnr.py "${output_dir}/dnr_v2"
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi
