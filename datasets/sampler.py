# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import random

import numpy as np
from torch.utils.data import BatchSampler, Sampler


class CustomSampler(Sampler):
    def __init__(self, dataset_length):
        self.dataset_length = dataset_length

    def __iter__(self):
        return iter(range(self.dataset_length))

    def __len__(self):
        return self.dataset_length


class CustomBatchSampler(BatchSampler):
    """A custom batch sampler that creates batches of indices with a specified
    number of sources and weights.

    Args:
        sampler (Sampler): The base sampler to use for generating indices.
        batch_size (int): The size of each batch.
        num_srcs_and_weights (dict): A dictionary where keys are the number of sources
            and values are their respective weights.
        shuffle (bool): Whether to shuffle the data before creating batches.
        drop_last (bool): Whether to drop the last incomplete batch.
    """

    def __init__(self, sampler, batch_size, num_srcs_and_weights, shuffle=True, drop_last=True):
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_srcs_and_weights = num_srcs_and_weights
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):

        indices = list(self.sampler)
        if self.shuffle:
            np.random.shuffle(indices)

        batches = []

        while len(indices) > 0:
            batch_indices = indices[: self.batch_size]
            indices = indices[self.batch_size :]
            num_src = random.choices(
                list(self.num_srcs_and_weights.keys()),
                weights=list(self.num_srcs_and_weights.values()),
            )[0]
            batch_indices = [f"{b}_{num_src}" for b in batch_indices]
            batches.append(batch_indices)

            if self.drop_last and len(indices) < self.batch_size:
                break

        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
