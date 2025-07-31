# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from fast_bss_eval import si_sdr_loss, si_sdr_pit_loss


def si_snr_loss(est, ref, solve_perm=True, clamp_db=70, **kwargs):
    assert est.ndim == ref.ndim == 3, (est.shape, ref.shape)

    # sometimes si_sdr_pit_loss causes an error in a single-speaker case
    # thus we use si_sdr_loss for such a case
    if est.shape[-2] == 1 or not solve_perm:
        return si_sdr_loss(est, ref, clamp_db=clamp_db)
    else:
        return si_sdr_pit_loss(est, ref, clamp_db=clamp_db)
