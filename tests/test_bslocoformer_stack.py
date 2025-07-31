# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch

from nets.bslocoformer_stack import BSLocoformerStack


@pytest.mark.parametrize("num_spk", [1, 2, 3, 4])
@pytest.mark.parametrize("n_first_blocks", [1, 4])
@pytest.mark.parametrize("n_second_blocks", [1, 2])
@pytest.mark.parametrize("emb_dim", [16])
@pytest.mark.parametrize("num_groups", [1, 8])
@pytest.mark.parametrize("tf_order", ["tf", "ft"])
@pytest.mark.parametrize("n_heads", [1, 4])
@pytest.mark.parametrize("attention_dim", [32])
@pytest.mark.parametrize("ffn_inner_dim", [32])
@pytest.mark.parametrize("conv1d_kernel", [4])
@pytest.mark.parametrize("conv1d_shift", [1])
@pytest.mark.parametrize("sample_rate", [48000])
@pytest.mark.parametrize("stft_size", [2048])
@pytest.mark.parametrize("eps", [1e-5])
def test_tuss_forward_backward(
    num_spk,
    n_first_blocks,
    n_second_blocks,
    # tf-locoformer related
    emb_dim,
    num_groups,
    tf_order,
    n_heads,
    attention_dim,
    ffn_inner_dim,
    conv1d_kernel,
    conv1d_shift,
    # banmd-split related
    sample_rate,
    stft_size,
    # others
    eps,
):
    config_first_block = dict(
        emb_dim=emb_dim,
        num_groups=num_groups,
        tf_order=tf_order,
        n_heads=n_heads,
        attention_dim=attention_dim,
        freq_ffn_config=[
            dict(
                conf=dict(
                    dim_inner=ffn_inner_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                )
            ),
            dict(
                conf=dict(
                    dim_inner=ffn_inner_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                )
            ),
        ],
        frame_ffn_config=[
            dict(
                conf=dict(
                    dim_inner=ffn_inner_dim,
                    conv1d_kernel=1,
                    conv1d_shift=conv1d_shift,
                )
            ),
            dict(
                conf=dict(
                    dim_inner=ffn_inner_dim,
                    conv1d_kernel=1,
                    conv1d_shift=conv1d_shift,
                )
            ),
        ],
    )

    config_second_block = dict(
        emb_dim=emb_dim,
        num_groups=num_groups,
        tf_order=tf_order,
        n_heads=n_heads,
        attention_dim=attention_dim,
        freq_ffn_config=[
            dict(
                conf=dict(
                    dim_inner=ffn_inner_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                )
            ),
            dict(
                conf=dict(
                    dim_inner=ffn_inner_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                )
            ),
        ],
        frame_ffn_config=[
            dict(
                conf=dict(
                    dim_inner=ffn_inner_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                )
            ),
            dict(
                conf=dict(
                    dim_inner=ffn_inner_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                )
            ),
        ],
    )

    # initialize TUSS model
    tuss = BSLocoformerStack(
        num_spk,
        config_first_block,
        config_second_block,
        n_first_blocks=n_first_blocks,
        n_second_blocks=n_second_blocks,
        sample_rate=sample_rate,
        stft_size=stft_size,
        eps=eps,
    )
    tuss.train()

    # Create dummy inputs
    n_batch = 2
    n_freqs = stft_size // 2 + 1
    n_frames = 50

    real = torch.randn(n_batch, n_frames, n_freqs)
    imag = torch.randn(n_batch, n_frames, n_freqs)
    x = torch.complex(real, imag)

    output = tuss(x)
    assert output.shape == (n_batch, num_spk, n_frames, n_freqs)
    sum(output).abs().mean().backward()
