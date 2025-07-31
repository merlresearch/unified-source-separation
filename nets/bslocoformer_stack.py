# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Dict

import torch
import torch.nn as nn
from packaging.version import parse as V
from rotary_embedding_torch import RotaryEmbedding

from .tuss import BandSplitModule, TFLocoformerBlock

is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")


class BSLocoformerStack(nn.Module):
    """TF-Locoformer model with the band-split encoder/decoder introduced in [1].

    This model is used as the `conventional model` in [1].
    To align with the structure of TUSS, this model is divided into two blocks, both of which
    consist of several stacked TF-Locoformer blocks.

    Parameters
    ----------
    num_spk: int
        Number of speakers (or sources) to separate.
    conf_first_block: Dict
        Configuration for the first block.
        See .yaml config files under ./configs directory for details.
    conf_second_block: Dict
        Configuration for the second block.
        See .yaml config files under ./configs directory for details.
    n_first_blocks: int
        Number of TF-Locoformer blocks in the first block.
    n_second_blocks: int
        Number of TF-Locoformer blocks in the second block.
    sample_rate: int
        Sample rate of the input audio.
    stft_size: int
        STFT size of the input audio.
    eps: float
        Small constant for normalization layer.

    References
    ----------
    [1]: Kohei Saijo, Janek Ebbers, François G Germain, Gordon Wichern, Jonathan Le Roux,
    “Task-Aware Unified Source Separation,” Proc. ICASSP,2025.
    """

    def __init__(
        self,
        num_spk: int,
        # general setup
        config_first_block: Dict,
        config_second_block: Dict,
        n_first_blocks: int = 4,
        n_second_blocks: int = 2,
        # band-split related
        sample_rate: int = 44100,
        stft_size: int = 2048,
        # others
        eps: float = 1.0e-5,
    ):
        super().__init__()
        assert is_torch_2_0_plus, "Support only pytorch >= 2.0.0"
        self._num_spk = num_spk

        self.first_block = nn.ModuleList([])
        rope_freq_first = RotaryEmbedding(config_first_block["attention_dim"] // config_first_block["n_heads"])
        rope_time_first = RotaryEmbedding(config_first_block["attention_dim"] // config_first_block["n_heads"])
        for _ in range(n_first_blocks):
            self.first_block.append(TFLocoformerBlock(rope_freq_first, rope_time_first, eps=eps, **config_first_block))

        # second block
        rope_freq_second = RotaryEmbedding(config_second_block["attention_dim"] // config_second_block["n_heads"])
        rope_time_second = RotaryEmbedding(config_second_block["attention_dim"] // config_second_block["n_heads"])

        self.second_block = nn.ModuleList([])
        for _ in range(n_second_blocks):
            self.second_block.append(
                TFLocoformerBlock(rope_freq_second, rope_time_second, eps=eps, **config_second_block)
            )

        self.emb_dim = emb_dim = config_first_block["emb_dim"]

        self.band_split_module = BandSplitModule(
            num_spk,
            emb_dim,
            stft_size,
            sample_rate,
        )

    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the BS-Locoformer model.

        Parameters
        ----------
        input: torch.Tensor, (n_batch, n_frames, n_freqs)
            Monaural complex tensor in STFT-domain
        prompts (List[List[str]]):
            List of prompt strings, where len(prompts) == n_batch and
            len(prompts[b]) == n_src for b in [0, n_batch-1].
            E.g., [["speech", "sfx", "sfx"], ["drums", "bass", "vocals"]].
            `n_src` can be variable in each forward pass but must be consistent in each batch.

        Returns
        ----------
        batch: torch.Tensor, (n_batch, n_src, n_frames, n_freqs)
            Separeted audio signals in TF-domain
        """
        assert input.ndim == 3, "Currently, only support monaural input."
        batch0 = input.unsqueeze(-1)
        batch = torch.cat((batch0.real, batch0.imag), dim=-1)  # [B, T, F, 2*M]
        n_batch, n_frames, n_freqs = batch.shape[:3]

        # normal spectrogram -> band-splitted tensor
        batch = self.band_split_module.band_split(batch)  # [B, -1, T, F]

        # separation
        for block in self.first_block:
            batch = block(batch)  # [B, -1, T, F]

        for block in self.second_block:
            batch = block(batch)

        # band-split tensor -> normal spectrogram
        batch = self.band_split_module.bandwise_decoding(batch)  # [B, n_srcs*2, T, F]
        batch = batch.view([n_batch, self.num_spk, 2, n_frames, n_freqs])

        batch = batch.to(torch.float32)

        # mapping or masking
        batch = torch.complex(batch[:, :, 0], batch[:, :, 1])
        batch = batch0.movedim(-1, -3) * batch

        return batch

    @property
    def num_spk(self):
        return self._num_spk
