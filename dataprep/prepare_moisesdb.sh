#!/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

output_dir=./moisesdb

if [ ! -e "${output_dir}/moisesdb/moisesdb_v0.1" ]; then
    echo "Please manually download moisesdb"
fi

if [ ! -f "${output_dir}/moisesdb_split.csv" ]; then
    wget -O "${output_dir}/moisesdb_split.csv" "https://raw.githubusercontent.com/kwatcharasupat/query-bandit/main/reproducibility/splits.csv"
fi

#################################
# Metadata preparation
#################################
filename="moisesdb_44.1k.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    python dataprep/moisesdb.py "${output_dir}/moisesdb/moisesdb_v0.1" "${output_dir}/moisesdb_split.csv"
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi


filename="moisesdb_44.1k_sad.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    # source activity detection to remove silent segments
    mkdir -p ./moisesdb_sad
    python dataprep/source_activity_detection.py \
	   --json_path datasets/json_files/moisesdb_44.1k.json \
	   --output_dir ./moisesdb_sad \
	   --segment_length 8 \
	   --min_power 1e-7 \
	   --min_thres 1e-5
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi
