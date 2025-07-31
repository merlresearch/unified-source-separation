#!/usr/bin/bash
# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Create symlinks for the data you already have

# DEMAND (write permission to this folder needed)
ln -s <path to parent folder of DKITCHEN, DLIVING, etc> demand
# DNR
ln -s <path to parent folder of dnr_v2> dnr
# DNS5 (write permission to this folder needed)
ln -s <path to parent folder of Track1_Headset> dns5_fullband
# FMA
ln -s <path to parent folder of fma_large, etc> fma
# FSD50K
ln -s <path to parent folder of FSD50K.metada, etc> fsd50k
# FUSS
ln -s <path to parent folder of fsd_data, etc> fuss
# MoisesDB
mkdir -p moisesdb
ln -s <path to parent folder of moisesdb_v0.1> moisesdb/moisesdb
# MUSDB18
ln -s <path to parent folder of musdbhq, etc> musdb18
# VCTK
ln -s <path to parent folder of VCTK-Corpus> vctk
# Voicebank_DEMAND
ln -s <path to parent folder of clean_testset_wav, etc> vctk_demand
# WHAMR
ln -s <path to parent folder of wav8k and wav16k> whamr
# WHAM_NOISE (write permission to this folder needed)
ln -s <path to parent folder of high_res_wham> wham_noise_48k
# WSJ0 and WSJ1
mkdir -p wsj
ln -s <path to parent folder of 11-1.1, etc> wsj/wsj0
ln -s <path to parent folder of 13-1.1, etc> wsj/wsj1
