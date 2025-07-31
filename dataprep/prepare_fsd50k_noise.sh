#!/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2024 Kohei Saijo
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0


# ported from https://github.com/urgent-challenge/urgent2025_challenge/blob/main/utils/prepare_FSD50K_noise.sh

output_dir=./fsd50k
mkdir -p "${output_dir}"

echo "=== Preparing FSD50K data ==="

#################################
# Download data
#################################
echo "Download FSD50K data"
if [ ! -e "${output_dir}/FSD50K.dev_audio" ]; then
    if [ ! -e "${output_dir}/unsplit.zip" ]; then
	org_dir=${PWD}
	cd ${output_dir}

	# download meta data
	wget -c -O FSD50K.ground_truth.zip https://zenodo.org/records/4060432/files/FSD50K.ground_truth.zip?download=1 && unzip FSD50K.ground_truth.zip
	wget -c -O FSD50K.metadata.zip https://zenodo.org/records/4060432/files/FSD50K.metadata.zip?download=1 && unzip FSD50K.metadata.zip
	wget -c -O FSD50K.doc.zip https://zenodo.org/records/4060432/files/FSD50K.doc.zip?download=1 && unzip FSD50K.doc.zip

	# download audio data
	wget -c -O FSD50K.dev_audio.zip https://zenodo.org/records/4060432/files/FSD50K.dev_audio.zip?download=1
	seq -w 1 5 | xargs -I {} -P 5 bash -c '
	        ii="{}"
		wget -c -O FSD50K.dev_audio.z0${ii} https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z0${ii}?download=1
		'

	# concat the zip file
	zip -s 0 FSD50K.dev_audio.zip --out unsplit.zip

	cd ${org_dir}
    fi

    unzip "${output_dir}/unsplit.zip" -d "${output_dir}"
fi

if [ ! -e "${output_dir}/FSD50K.eval_audio" ]; then
    if [ ! -e "${output_dir}/unsplit_eval.zip" ]; then
	org_dir=${PWD}
	cd ${output_dir}

	wget -c -O FSD50K.eval_audio.z01 https://zenodo.org/records/4060432/files/FSD50K.eval_audio.z01?download=1
	wget -c -O FSD50K.eval_audio.zip https://zenodo.org/records/4060432/files/FSD50K.eval_audio.zip?download=1

	zip -s 0 FSD50K.eval_audio.zip --out unsplit_eval.zip

	cd ${org_dir}
    fi

    unzip "${output_dir}/FSD50K.eval_audio.zip" -d "${output_dir}"
fi

if [ ! -e "./dataprep/datafiles/ontology" ]; then
    # download AudioSet ontology"
    git clone https://github.com/audioset/ontology.git "./dataprep/datafiles/ontology"
fi

#################################
# Metadata preparation
#################################
echo "Metadata preparation for FSD50K"

filename="fsd50k_44.1k.json"
if [ ! -f "./datasets/json_files/${filename}" ]; then
    python dataprep/fsd50k.py \
	   --data_dir "${output_dir}" \
	   --ontology_path "./dataprep/datafiles/ontology/ontology.json"
else
    echo "./datasets/json_files/${filename} already exists, delete to regenerate"
fi
