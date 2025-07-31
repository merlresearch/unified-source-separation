# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import torch

from nets.tuss import TussModel


@pytest.mark.parametrize(
    "prompts",
    [
        ["speech"],
        ["speech", "musicbg"],
        ["speech", "sfxbg"],
        ["sfx", "sfx", "sfx"],
        ["bass", "drums", "vocals", "other"],
    ],
)
@pytest.mark.parametrize("nblocks_cross_prompt_module", [1, 4])
@pytest.mark.parametrize("nblocks_cond_tse_module", [1, 2])
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
@pytest.mark.parametrize("prompt_size", [1])
@pytest.mark.parametrize("use_sos_token", [True, False])
def test_tuss_forward_backward(
    prompts,
    nblocks_cross_prompt_module,
    nblocks_cond_tse_module,
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
    prompt_size,
    use_sos_token,
):
    conf_cross_prompt_module = dict(
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

    conf_cond_tse_module = dict(
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
    prompts_dedup = list(set(prompts))
    tuss = TussModel(
        prompts_dedup,
        conf_cross_prompt_module,
        conf_cond_tse_module,
        nblocks_cross_prompt_module=nblocks_cross_prompt_module,
        nblocks_cond_tse_module=nblocks_cond_tse_module,
        sample_rate=sample_rate,
        stft_size=stft_size,
        eps=eps,
        prompt_size=prompt_size,
        use_sos_token=use_sos_token,
    )
    tuss.train()

    # Create dummy inputs
    n_batch = 2
    n_freqs = stft_size // 2 + 1
    n_frames = 50

    prompts = [prompts for _ in range(n_batch)]

    real = torch.randn(n_batch, n_frames, n_freqs)
    imag = torch.randn(n_batch, n_frames, n_freqs)
    x = torch.complex(real, imag)

    output = tuss(x, prompts)
    assert output.shape == (n_batch, len(prompts[0]), n_frames, n_freqs)
    sum(output).abs().mean().backward()
