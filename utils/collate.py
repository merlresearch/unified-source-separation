# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import collections.abc as container_abcs

import torch
import torch.nn.utils.rnn as rnn


def collate_seq(batch):
    elem = batch[0]
    elem_type = type(elem)
    if elem_type.__name__ == "ndarray":
        pad_features = rnn.pad_sequence([torch.tensor(b, dtype=torch.float32) for b in batch], batch_first=True)
        return pad_features

    elif isinstance(elem, torch.Tensor):
        pad_features = rnn.pad_sequence([b for b in batch], batch_first=True)
        if pad_features.ndim == 3:
            pad_features = pad_features.transpose(-1, -2)  # n_src <--> n_samples
        return pad_features

    elif isinstance(elem, tuple):
        transposed = zip(*batch)
        return [collate_seq(samples) for samples in transposed]

    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_seq([d[key] for d in batch]) for key in elem}

    else:
        # for other stuff just return it and don't collate
        return [b for b in batch]
