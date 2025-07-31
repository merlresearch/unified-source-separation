# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import itertools

import torch


def snr_loss(est, ref, solve_perm=True, return_perm=False, eps=1.0e-8, **kwargs):
    nsrc_est = est.shape[1]
    nsrc_ref = ref.shape[1]

    ref_power = (ref**2).sum(dim=-1)  # (n_batch, n_src)

    # permutations
    if solve_perm:
        perms = torch.LongTensor(list(itertools.permutations(torch.arange(nsrc_est), nsrc_ref)))
    else:
        # fixed permutation
        perms = torch.LongTensor(list(range(nsrc_ref))).unsqueeze(0)

    # compute soft threshold beforehand
    snrs, snrs_average = [], []
    for perm in perms:
        est_permed = est[..., perm, :]  # permuted sounds
        assert est_permed.shape == ref.shape, (est_permed.shape, ref.shape)

        noise = est_permed - ref
        noise_power = (noise**2).sum(dim=-1)

        # denominator of snr (n_batch)
        snr = 10 * (torch.log10(noise_power + eps) - torch.log10(ref_power + eps))
        snrs.append(snr)
        snrs_average.append(snr.mean(dim=-1))

    # get the best loss
    snrs = torch.stack(snrs, dim=-1)  # (n_batch, n_src, n_perms)
    snrs_average = torch.stack(snrs_average, dim=-1)  # (n_batch, n_perms)
    ll, indices = torch.min(snrs_average, dim=-1)
    expanded_indices = indices.unsqueeze(-1).unsqueeze(-1).expand(snrs.shape[0], nsrc_ref, 1)
    loss = torch.gather(snrs, -1, expanded_indices).squeeze(-1)

    if not return_perm:
        return loss

    # get best permutation
    best_perms = []
    for idx in indices:
        best_perms.append(perms[idx])
    best_perms = torch.stack(best_perms, dim=0)
    return loss, best_perms


def snr_with_zeroref_loss(
    est, ref, n_src, snr_max=30, zero_ref_loss_weight=0.1, solve_perm=True, return_perm=False, eps=1.0e-7, **kwargs
):
    mix = ref.sum(dim=1, keepdim=True)

    # if reference is smaller than expected,
    # zero signal is added
    if ref.shape[1] < n_src:
        zeros = ref.new_zeros(ref.shape[0], n_src - ref.shape[1], ref.shape[2])
        ref = torch.cat((ref, zeros), dim=1)

    ref_power = (ref**2).sum(dim=-1)  # (n_batch, n_src)
    mix_power = (mix**2).sum(dim=-1).tile(1, ref_power.shape[-1])

    # 0 when reference is zero else 1
    coef = ref_power > 0.0

    # permutations
    if solve_perm:
        perms = torch.LongTensor(list(itertools.permutations(range(n_src))))
    else:
        # fixed permutation
        perms = torch.LongTensor(list(range(n_src))).unsqueeze(0)

    # compute soft threshold beforehand
    regularizer = coef * ref_power + ~coef * mix_power
    soft_thres = (10 ** (-snr_max / 10)) * regularizer
    snrs, snrs_average = [], []
    for perm in perms:
        assert est.shape[-2] == perm.shape[-1] == n_src, (n_src, est.shape, perm.shape)
        est_permed = est[..., perm, :]  # permuted sounds
        noise = est_permed - ref
        noise_power = (noise**2).sum(dim=-1)

        # denominator of snr (n_batch)
        snr = 10 * torch.log10(noise_power + soft_thres + eps)

        # make loss the same as SNR computation
        # by dividing lossses of active source by reference power
        denom = 10 * torch.log10(regularizer + eps)
        snr = snr - denom * coef  # multiply coef to ignore espcilon

        # loss weighting on zero-reference loss
        # high weight may make training unstable
        if zero_ref_loss_weight != 1.0:
            assert snr.shape == coef.shape, (snr.shape, coef.shape)
            snr = snr * (coef + ~coef * zero_ref_loss_weight)

        snrs.append(snr)
        snrs_average.append(snr.mean(dim=-1))

    # get the best loss
    snrs = torch.stack(snrs, dim=-1)  # (n_batch, n_src, n_perms)
    snrs_average = torch.stack(snrs_average, dim=-1)  # (n_batch, n_perms)
    ll, indices = torch.min(snrs_average, dim=-1)
    expanded_indices = indices.unsqueeze(-1).unsqueeze(-1).expand(snrs.shape[0], n_src, 1)
    loss = torch.gather(snrs, -1, expanded_indices).squeeze(-1)

    if not return_perm:
        return loss

    # # get best permutation
    best_perms = []
    for idx in indices:
        best_perms.append(perms[idx])
    best_perms = torch.stack(best_perms, dim=0)
    return loss, best_perms
