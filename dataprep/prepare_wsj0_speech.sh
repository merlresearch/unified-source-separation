#!/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

output_dir=./wsj

echo "=== Preparing WSJ0 data ==="

#################################
# Download data
#################################
if [ ! -d "${output_dir}/wsj0" ]; then
    echo "Please manually download the data from https://catalog.ldc.upenn.edu/LDC93s6a and save them under the directory '$output_dir/wsj0'"
fi

#################################
# Convert sph formats to wav
#################################
if [[ ! -f "${output_dir}/wsj_sph2wav.done" && ! -e "${output_dir}/wsj0_wav" ]]; then
    #################################
    # Download sph2pipe
    #################################
    if ! command -v sph2pipe; then
        echo "Installing sph2pipe from https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools"
        SPH2PIPE_VERSION=2.5

        if [ ! -e sph2pipe_v${SPH2PIPE_VERSION}.tar.gz ]; then
            wget -nv -T 10 -t 3 -O sph2pipe_v${SPH2PIPE_VERSION}.tar.gz \
                "https://github.com/burrmill/sph2pipe/archive/refs/tags/${SPH2PIPE_VERSION}.tar.gz"
        fi

        if [ ! -e sph2pipe-${SPH2PIPE_VERSION} ]; then
            tar --no-same-owner -xzf sph2pipe_v${SPH2PIPE_VERSION}.tar.gz
            rm -rf sph2pipe 2>/dev/null || true
            ln -s sph2pipe-${SPH2PIPE_VERSION} sph2pipe
        fi

        make -C sph2pipe
        sph2pipe=$PWD/sph2pipe/sph2pipe
    else
        sph2pipe=sph2pipe
    fi

    echo "[WSJ] converting sph audios to wav"
    find "${output_dir}/wsj0/" -iname '*.wv1' | while read -r fname; do
        # It takes ~23 minutes to finish audio format conversion in "${output_dir}/wsj0_wav"
        fbasename=$(basename "${fname}" | sed -e 's/\.wv1$//i')
        fdir=$(realpath --relative-to="${output_dir}/wsj0/" $(dirname "${fname}"))
        out="${output_dir}/wsj0_wav/${fdir}/${fbasename}.wav"
        mkdir -p "${output_dir}/wsj0_wav/${fdir}"
        "${sph2pipe}" -f wav "${fname}" > "${out}"
    done
    touch ${output_dir}/wsj_sph2wav.done
else
    echo "[WSJ] sph format conversion already finished"
fi

#################################
# Metadata preparation
#################################
echo "Metadata preparation for WSJ0"

if [ ! -d "dataprep/datafiles/create-speaker-mixtures" ]; then
    if [ ! -3 "dataprep/datafiles/create-speaker-mixtures.zip" ]; then
	wget -O "dataprep/datafiles/create-speaker-mixtures.zip" "https://www.merl.com/research/highlights/deep-clustering/create-speaker-mixtures.zip"
    fi
    unzip "dataprep/datafiles/create-speaker-mixtures.zip" -d "dataprep/datafiles/create-speaker-mixtures"
fi


filename="wsj0_speech_16k.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    python dataprep/wsj0.py \
	   --speech_dir ${output_dir}/wsj0_wav \
	   --speech_metadata_dir dataprep/datafiles/create-speaker-mixtures
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi
