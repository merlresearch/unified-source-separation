# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections import defaultdict
from functools import partial

import torch

from .si_snr import si_snr_loss
from .snr import snr_loss, snr_with_zeroref_loss


class LossWrapper:
    def __init__(self, loss_func_name, loss_func_conf, pit_loss=False):
        if loss_func_name == "si_snr":
            loss_func = si_snr_loss
        elif loss_func_name == "snr":
            loss_func = snr_loss
        elif loss_func_name == "snr_with_zero_refs":
            loss_func = snr_with_zeroref_loss
        else:
            raise NotImplementedError()

        # NOTE: assuming that permutation is solved in self.loss_func
        self.loss_func = partial(loss_func, **loss_func_conf)
        self.pit_loss = pit_loss

    @torch.cuda.amp.autocast(enabled=False)
    def __call__(self, est, ref, prompts, return_list=False):
        if self.pit_loss:
            # compute PIT loss, used to train a conventional model
            loss = self._comptute_pit_loss(est, ref, return_list=return_list)
        else:
            # compute PIT loss for each prompt, used to train the TUSS model
            loss, prompts = self._compute_loss_for_each_prompt(est, ref, prompts, return_list)
        return loss, prompts

    def _comptute_pit_loss(self, est, ref, return_list=False):
        # unlike _compute_loss_for_each_prompt, loss is computed regardless of prompts
        # NOTE: assuming that permutation is solved in self.loss_func
        m = min(est.shape[-1], ref.shape[-1])
        est, ref = est[..., :m], ref[..., :m]
        loss = self.loss_func(est, ref, n_src=est.shape[-2])

        if return_list:
            loss = loss.cpu().numpy()
            loss_list = []
            for b in range(len(loss)):
                loss_list.append(loss[b].tolist())
            return loss_list
        else:
            return loss.mean()

    def _compute_loss_for_each_prompt(self, est, ref, prompts, return_list=False):
        # compute PIT loss for each prompt
        # NOTE: assuming that permutation is solved in self.loss_func
        n_batch, n_src = est.shape[:2]
        est_list, ref_list, prompts = self._make_lists_for_each_prompt(est, ref, prompts)

        loss = 0.0 if not return_list else []

        for b, (est_b, ref_b) in enumerate(zip(est_list, ref_list)):
            if isinstance(loss, list):
                loss.append([])
            for est_l, ref_l in zip(est_b, ref_b):
                m = min(est_l.shape[-1], ref_l.shape[-1])
                est_l, ref_l = est_l[..., :m], ref_l[..., :m]
                assert est_l.shape == ref_l.shape, (est_l.shape, ref_l.shape)

                # loss computation
                if return_list:
                    loss_tmp = (self.loss_func(est_l, ref_l, n_src=est_l.shape[-2]).cpu().numpy())[0]
                    loss[b].extend(loss_tmp.tolist())
                else:
                    loss += self.loss_func(est_l, ref_l, n_src=est_l.shape[-2]).sum()

        if not return_list:
            loss /= n_src * n_batch
        return loss, prompts

    def _make_lists_for_each_prompt(self, est, ref, prompts):
        # prompts are randomly selected, so sort them
        # to compute PIT loss for intra-prompts but not inter-prompts
        est_list, ref_list = [], []
        for b in range(len(prompts)):
            index_dict = defaultdict(list)
            est_list.append([])
            ref_list.append([])
            for i, pr in enumerate(prompts[b]):
                index_dict[pr].append(i)
                indices = list(index_dict.values())

            est_list[b].extend([est[b, [idx]] for idx in indices])
            ref_list[b].extend([ref[b, [idx]] for idx in indices])

        return est_list, ref_list, prompts
