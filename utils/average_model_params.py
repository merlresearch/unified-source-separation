# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2025 ESPnet Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict

import torch


@torch.no_grad()
def average_model_params(
    model_paths: Path,
) -> Dict:
    """Return an averaged model parameters from the given model directory

    Args:
        output_dir: The directory contains the model file for each epoch
        reporter: Reporter instance
        best_model_criterion: Give criteria to decide the best model.
            e.g. [("valid", "loss", "min"), ("train", "acc", "max")]
        nbest: Number of best model files to be averaged
        suffix: A suffix added to the averaged model file name
    """
    epochs = len(model_paths)

    print(f"Average these {epochs} checkpoints:")

    _loaded = {}
    avg = None
    for e, model_path in enumerate(model_paths):
        if e not in _loaded:
            print(model_path)
            _loaded[e] = torch.load(
                model_path,
                map_location="cpu",
            )
            if model_path.suffix == ".ckpt":
                _loaded[e] = _loaded[e]["state_dict"]
        states = _loaded[e]

        if avg is None:
            avg = states
        else:
            # Accumulated
            for k in avg:
                avg[k] = avg[k] + states[k]
    for k in avg:
        if str(avg[k].dtype).startswith("torch.int"):
            # For int type, not averaged, but only accumulated.
            # e.g. BatchNorm.num_batches_tracked
            # (If there are any cases that require averaging
            #  or the other reducing method, e.g. max/min, for integer type,
            #  please report.)
            pass
        else:
            avg[k] = avg[k] / epochs

    return avg
