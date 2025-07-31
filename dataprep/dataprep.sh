#!/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

datasets=(
    "vctk_speech"
    "demand_noise"
    "vctk_demand"
    "wsj0_speech"
    "wham_noise"
    "whamr"
    "librivox_urgent2024"
    "fsd50k_noise"
    "fma_noise"
    "dnr"
    "fuss"
    "musdb"
    "moisesdb"
)

mkdir -p "./datasets/json_files"
for dataset in ${datasets[@]}; do
    . "dataprep/prepare_${dataset}.sh"
done
