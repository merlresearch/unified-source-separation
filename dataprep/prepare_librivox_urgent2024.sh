#!/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright 2024 Wangyou Zhang
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0

# copied from https://github.com/urgent-challenge/urgent2025_challenge/blob/main/utils/prepare_DNS5_librivox_speech.sh

output_dir=./dns5_fullband # please do not change this path not to break the consistency with the provided BW_EST_FILE
mkdir -p ${output_dir}

echo "=== Preparing DNS5 LibriVox speech data ==="

#################################
# Download data
#################################
# Refer to https://github.com/microsoft/DNS-Challenge/blob/master/download-dns-challenge-5-headset-training.sh
# this will take some time to download
if [ ! -e "${output_dir}/Track1_Headset" ]; then
    ./dataprep/download_librivox_speech.sh ${output_dir} 8
fi

#################################
# Data preprocessing
#################################
BW_EST_FILE=dataprep/datafiles/dns5_clean_read_speech.json
BW_EST_FILE_JSON_GZ="dataprep/datafiles/dns5_clean_read_speech.json.gz"
if [ ! -f ${BW_EST_FILE} ]; then
    gunzip -c $BW_EST_FILE_JSON_GZ > $BW_EST_FILE
fi

RESAMP_SCP_FILE=dataprep/datafiles/dns5_clean_read_speech_resampled.scp
if [ ! -f ${RESAMP_SCP_FILE} ]; then
    echo "[DNS5 LibriVox] resampling to estimated audio bandwidth"
    OMP_NUM_THREADS=1 python dataprep/resample_to_estimated_bandwidth.py \
        --bandwidth_data ${BW_EST_FILE} \
        --out_scpfile ${RESAMP_SCP_FILE} \
        --outdir "${output_dir}/Track1_Headset/resampled/clean/read_speech" \
        --max_files 5000 \
        --nj 8 \
        --chunksize 1000
else
    echo "Resampled scp file already exists. Delete ${RESAMP_SCP_FILE} if you want to re-resample."
fi

#########################################
# Data filtering based on VAD and DNSMOS
#########################################
DNSMOS_JSON_FILE="dataprep/datafiles/dns5_clean_read_speech_resampled_dnsmos.json"
DNSMOS_GZ_FILE="dataprep/datafiles/dns5_clean_read_speech_resampled_dnsmos.json.gz"
if [ -f ${DNSMOS_GZ_FILE} ]; then
    gunzip -c ${DNSMOS_GZ_FILE} > ${DNSMOS_JSON_FILE}
fi

# remove non-speech samples
VAD_SCP_FILE="dataprep/datafiles/dns5_clean_read_speech_resampled_filtered_vad.scp"
if [ ! -f ${VAD_SCP_FILE} ]; then
    echo "[DNS5 LibriVox] filtering via VAD"
    OMP_NUM_THREADS=1 python dataprep/filter_via_vad.py \
        --scp_path ${RESAMP_SCP_FILE} \
        --outfile ${VAD_SCP_FILE} \
        --vad_mode 2 \
        --threshold 0.2 \
        --nj 8 \
        --chunksize 200
else
    echo "VAD scp file already exists. Delete ${VAD_SCP_FILE} if you want to re-estimate."
fi

# remove low-quality samples
FILTERED_SCP_FILE="dataprep/datafiles/dns5_clean_read_speech_resampled_filtered_dnsmos.scp"
if [ ! -f ${FILTERED_SCP_FILE} ]; then
    echo "[DNS5 LibriVox] filtering via DNSMOS"
    python dataprep/filter_via_dnsmos.py \
        --scp_path "${VAD_SCP_FILE}" \
        --json_path "${DNSMOS_JSON_FILE}" \
        --outfile ${FILTERED_SCP_FILE} \
        --score_name BAK --threshold 3.0
else
    echo "Filtered scp file already exists. Delete ${FILTERED_SCP_FILE} if you want to re-estimate."
fi

FILTERED_SCP_FILE="dataprep/datafiles/dns5_clean_read_speech_resampled_filtered_dnsmos.scp"
#################################
# Metadata preparation
#################################
echo "Metadata preparation for DNS5 LibriVox"

filename="librivox_urgent2024.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    python dataprep/librivox_urgent2024.py \
	   --speech_scp "${FILTERED_SCP_FILE}"
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi
