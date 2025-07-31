# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import math

import torch


def get_grad_norm(parameters, norm_type=2.0, device=None):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if device is None:
        device = parameters[0].device
    if norm_type == math.inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        out = torch.empty(len(parameters), device=device)
        for i, p in enumerate(parameters):
            torch.norm(p.grad.data.to(device), norm_type, out=out[i])
        total_norm = torch.norm(out, norm_type)
    return total_norm.item()
