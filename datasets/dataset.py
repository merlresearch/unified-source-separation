# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F


class DynamicMixingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        partition: str,
        json_paths: List[Path] = None,
        chunk_size: int | None = None,
        sample_rate: int | None = None,
        downsample_to_minimum_fs: bool = True,
        prompts: List[str] = ["speech", "music", "sfx"],
        max_num_src: int = 4,
        num_data_per_epoch: int | None = None,
    ):
        """Dataset class for dynamic mixing. Used in the training stage.
        Mixtures are made in a following way:
        1. Randomly select prompts.
        2. Randomly load sources based on the prompts.
        3. Adjust the energy of the sources based on the first source.
        4. Mix the sources.

        Parameters
        ----------
        partition: str
            Dataset partition. Must be "train".
        json_paths: List[Path]
            List of the paths to the json files. Defined in the config (.yaml) file.
        chunk_size: int
            Input length in seconds during training.
        sample_rate: int
            If given, audios are always resampled to this rate AFTER mixing.
        downsample_to_minimum_fs: bool
            Whether to downsample to the minimum sampling rate BEFORE mixing.
            Different audios can have different sampling rate when using multiple datasets.
            If the sampling rates are 16kHz, 22.5kHz, and 44.1kHz, these three audios
            are resampled to 16kHz.
        prompts: List[str]
            List of prompts. Defined in the config (.yaml) file.
        max_num_src: int
            Maxium number of sources in a mixture INCLUDEING additional noise.
        """
        assert partition == "train", partition
        self.training = partition == "train"

        self.data_copy = OrderedDict()
        for json_path in json_paths:
            with open(json_path, "r") as f:
                data = json.load(f)[partition]
                for prompt, sound_data in data.items():
                    if prompt not in self.data_copy:
                        self.data_copy[prompt] = []
                    self.data_copy[prompt].extend(sound_data)

        self.chunk_size = chunk_size if partition == "train" else None

        self.prompts = list(prompts.keys())
        self.prompt_init_prob = [prompts[prompt]["init_prob"] for prompt in self.prompts]
        self.prompt_metadata = prompts

        for prompt, prompt_next_prob in self.prompt_metadata.items():
            assert list(prompt_next_prob["next"].keys()) == self.prompts, (
                prompt,
                prompt_next_prob["next"].keys(),
                self.prompts,
            )

        self.sample_rate = sample_rate
        self.downsample_to_minimum_fs = downsample_to_minimum_fs

        self.max_num_src = max_num_src

        self.num_data_per_epoch = num_data_per_epoch

        self.setup()

    def setup(self):
        # shuffle the data
        # called at the beginning of each epoch in lightning_train.py
        self.data = {}
        for prompt in self.data_copy:
            self.data[prompt] = random.sample(self.data_copy[prompt], len(self.data_copy[prompt]))

    def get_len(self):
        return self.__len__()

    def __len__(self):
        if self.num_data_per_epoch is not None:
            return self.num_data_per_epoch
        return max([len(v) for v in self.data.values()])

    def __getitem__(self, index):
        # index includes index and number of prompts
        # in the form of {index}_{num_prompts}
        index, num_prompts = index.split("_")
        mix, ref, prompts = self._dynamic_mixing(int(index), int(num_prompts))

        return mix, ref, prompts

    def _dynamic_mixing(self, index, num_prompts):
        # randomly select prompts
        prompts = self._select_prompts(num_prompts)
        while prompts is None:
            prompts = self._select_prompts(num_prompts)

        # load sources based on the prompts
        ref = self._load_sources_with_prompts(prompts)

        # adjust the energy of the sources based on the first source
        ref = self._adjust_audio_scale_with_random_gain(ref, prompts=prompts)

        # dynamic mixing
        mix = torch.stack(ref, dim=0).sum(dim=0)

        # discard additional noise and prompts for the final output
        ref, prompts = ref[:num_prompts], prompts[:num_prompts]
        ref = torch.stack(ref, dim=0)

        return mix, ref.T, prompts

    def _select_prompts(self, num_prompts, prompts=[], allow_duplication=True):
        for i in range(len(prompts), num_prompts):
            if i == 0:
                weights = self.prompt_init_prob
                prompts = random.choices(self.prompts, weights=weights, k=1)
            else:
                # select next prompt based on the previous prompt
                prev_prompt = prompts[-1]
                weights = list(self.prompt_metadata[prev_prompt]["next"].values())

                black_list = [
                    [pp for pp in list(self.prompt_metadata[p]["next"]) if self.prompt_metadata[p]["next"][pp] == 0.0]
                    for p in prompts
                ]
                black_list = list(set([item for sublist in black_list for item in sublist]))

                if not allow_duplication:
                    black_list = list(set(black_list + prompts))

                # choose one prompt
                p = random.choices(self.prompts, weights=weights, k=1)[0]

                # some prompts are not allowed to appear multiple times
                count = 0
                while sum(1 for pp in prompts if pp == p) >= self.prompt_metadata[p]["num_appear"] or p in black_list:
                    p = random.choices(self.prompts, weights=weights, k=1)[0]
                    count += 1
                    if count == 100:
                        return None
                prompts.append(p)
        return prompts

    def _load_sources_with_prompts(self, prompts):
        ref = []

        past_indices = []
        orig_sample_rates = []
        for i, prompt in enumerate(prompts):
            # if prompt is xxx-mix, either randomly select 1-3 sources and mix them
            # or randomly pick a sample from an already mixed dataset.
            # for music-mix, we use FMA as the mixed dataset.

            if prompt.endswith("bg"):
                # pick music background
                p_dynamic_mixing = self.prompt_metadata[prompt]["dynamic_mixing_prob"]
                if random.random() > p_dynamic_mixing:
                    audio, past_indices, fs = self._load_nonzero_audio(
                        prompt,
                        past_indices=past_indices,
                    )
                    ref.append(audio)
                    orig_sample_rates.append(fs)
                # dynamic mixing for making a mixture of intruments
                else:
                    # randomly choose the number of sources
                    nsrcs_weights = self.prompt_metadata[prompt]["num_srcs_and_weights"]
                    num_src = random.choices(
                        list(nsrcs_weights.keys()),
                        weights=list(nsrcs_weights.values()),
                        k=1,
                    )[0]

                    ref_tmp = []
                    orig_sample_rates_tmp = []
                    if prompt == "musicbg":
                        prompts = ["bass", "drums", "vocals", "other"]
                        prompts = random.sample(prompts, k=num_src)  # not allow duplicates

                        for prompt in prompts:
                            audio, past_indices, fs = self._load_nonzero_audio(
                                prompt,
                                past_indices=past_indices,
                            )
                            ref_tmp.append(audio)
                            orig_sample_rates_tmp.append(fs)
                    else:
                        prompt = prompt.replace("bg", "")
                        prompts = [prompt] * num_src

                        ref_tmp = []
                        for _ in range(num_src):
                            audio, past_indices, fs = self._load_nonzero_audio(
                                prompt,
                                past_indices=past_indices,
                            )

                            ref_tmp.append(audio)
                            orig_sample_rates_tmp.append(fs)

                    # resample to minimum sampling rate
                    min_fs = min(orig_sample_rates_tmp)
                    orig_sample_rates.append(min_fs)
                    ref_tmp = [F.resample(r, fs, min_fs) for r, fs in zip(ref_tmp, orig_sample_rates_tmp)]
                    ref_tmp = [self._zero_pad(r, min_fs) for r in ref_tmp]

                    # adjust the gain
                    ref_tmp = self._adjust_audio_scale_with_random_gain(ref_tmp, prompts=prompts)
                    ref.append(torch.stack(ref_tmp, dim=0).sum(dim=0))

            else:
                audio, past_indices, fs = self._load_nonzero_audio(
                    prompt,
                    past_indices=past_indices,
                )
                orig_sample_rates.append(fs)

                ref.append(audio)

        # resample to minimum sampling rate and then up-sample to self.sample_rate
        if self.downsample_to_minimum_fs:
            min_fs = min(orig_sample_rates)
        else:
            min_fs = None
        ref = [self._resample_audio(r, fs, min_fs)[0] for r, fs in zip(ref, orig_sample_rates)]

        # zero-padding if necessary to adjust the length
        ref = [self._zero_pad(r, self.sample_rate) for r in ref]

        return ref

    def _load_nonzero_audio(self, prompt, idx=None, past_indices=[], **kwargs):
        if idx is None:
            idx = random.randint(0, len(self.data[prompt]) - 1)
        audio, fs = self._read_audio(self.data[prompt][idx])
        while abs(audio).sum() == 0.0 or idx in past_indices:
            idx = random.randint(0, len(self.data[prompt]) - 1)
            audio, fs = self._read_audio(self.data[prompt][idx])
        past_indices.append(idx)

        return audio, past_indices, fs

    def _adjust_audio_scale_with_random_gain(self, ref, prompts):
        # first, normalize each audio signal
        ref = [r / torch.sqrt((r**2).mean()) if abs(r).sum() != 0.0 else r for r in ref]

        # adjust gain
        for i in range(len(ref)):
            gain_range = self.prompt_metadata[prompts[i]]["gain"]

            if isinstance(gain_range, list):
                gain_db = random.uniform(gain_range[0], gain_range[1])
            else:
                gain_db = gain_range

            gain = 10 ** (gain_db / 20)
            ref[i] *= gain
        return ref

    def _resample_audio(self, audio, orig_fs, resample_rate):
        fs = orig_fs
        if resample_rate is not None and fs != resample_rate:
            audio = F.resample(audio, fs, resample_rate)
            fs = resample_rate

        if self.sample_rate is not None and fs != self.sample_rate:
            audio = F.resample(audio, fs, self.sample_rate)
            fs = self.sample_rate

        return audio, fs

    def _zero_pad(self, audio, fs):
        chunk_size = fs * self.chunk_size
        if audio.shape[0] < chunk_size:
            pad = chunk_size - audio.shape[0]
            left_pad = random.randint(0, pad)
            right_pad = pad - left_pad
            audio = torch.cat([torch.zeros(left_pad), audio, torch.zeros(right_pad)])
        return audio

    def _read_audio(self, data):
        orig_sample_rate = data["sample_rate"]
        num_samples = data["num_samples"]

        if self.chunk_size is None:
            start, stop = 0, -1
        else:
            chunk_size = orig_sample_rate * self.chunk_size
            if chunk_size < num_samples:
                start = np.random.randint(0, num_samples - chunk_size)
                stop = start + chunk_size
            else:
                start, stop = 0, -1

        try:
            ref, fs = torchaudio.load(
                data["path"],
                frame_offset=start,
                num_frames=stop - start,
                backend="soundfile",
            )
        except Exception as e:
            print(f"Error in loading {data['path']}: {e}", flush=True)
            return self._read_audio(data)

        assert fs == orig_sample_rate, (orig_sample_rate, fs)

        if ref.ndim == 2:  # multi-channel -> monaural
            ref = ref[0]

        return ref, orig_sample_rate


class FixedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        partition: str,
        json_path: Path | None = None,
        num_data: int | None = None,
        sample_rate: int | None = None,
        ref_channel: int = 0,
    ):
        """Dataset class without dynamic mixing. Used in the validation and test stages.

        Parameters
        ----------
        path_list: List[Dict]
            List of the metadata of each audio data.
        training: bool
            Whether or not this dataset is initialized for training or not.
        sample_rate: int
            Sampling rate.
        chunk_size: int or float
            Input length in seconds during training.
        normalization: bool
            Whether to apply the variance normalization.
        """

        assert partition in ["train", "valid", "test"], partition
        self.training = partition == "train"

        data = []
        with open(json_path, "r") as f:
            j = json.load(f)[partition]
            for value in j.values():
                data.append(value)

        if num_data is not None:
            data = data[:num_data]

        # Sort data by length in validation stage to reduce
        # the impact of zero-padding happening when making a mini-batch.
        if partition == "valid":
            self.data = sort_dicts_by_key(data, "num_samples")
        else:
            self.data = data

        self.sample_rate = sample_rate
        self.ref_channel = ref_channel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self._read_audio(index)

    def _resample_audio(self, audio, orig_fs, resample_rate):
        fs = orig_fs
        if resample_rate is not None and fs != resample_rate:
            audio = F.resample(audio, fs, resample_rate)
            fs = resample_rate

        if self.sample_rate is not None and fs != self.sample_rate:
            audio = F.resample(audio, fs, self.sample_rate)
            fs = self.sample_rate

        return audio, fs

    def _read_audio(self, index):
        data = self.data[index]
        orig_sample_rate = data["sample_rate"]

        # load reference audios
        prompts, ref = [], []
        for key, ref_path in data["ref"].items():
            prompts.append(key.split("_")[0])  # e.g., speech_1 -> speech
            try:
                r, fs = torchaudio.load(
                    ref_path,
                    backend="soundfile",
                )
            except Exception as e:
                print(f"Error in loading {ref_path}: {e}", flush=True)
                continue
            assert fs == orig_sample_rate, (orig_sample_rate, fs)

            # multi-channel -> monaural
            if r.ndim == 2:
                r = r[self.ref_channel]

            r, fs = self._resample_audio(r, fs, resample_rate=None)
            ref.append(r)

        ref = np.stack(ref, axis=-2)  # (n_chan, n_src, chunk_size)
        ref = torch.from_numpy(ref)
        mix = ref.sum(dim=-2)

        return (
            mix,
            ref.T,
            prompts,
            data["dataset_name"],
            orig_sample_rate,
        )


def sort_dicts_by_key(dicts, key):
    # Sort the list of dictionaries by the specified key
    sorted_dicts = sorted(dicts, key=lambda d: d[key], reverse=True)
    return sorted_dicts
