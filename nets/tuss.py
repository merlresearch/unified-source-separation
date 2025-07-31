# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from itertools import accumulate
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as V
from rotary_embedding_torch import RotaryEmbedding
from torch.nn.attention import SDPBackend, sdpa_kernel

is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")

# same configuration as BS-Roformer
# https://arxiv.org/abs/2309.02612
BAND_SPLIT = {
    # (frequency range): num_bins
    (0, 1000): 2,
    (1000, 2000): 4,
    (2000, 4000): 12,
    (4000, 8000): 24,
    (8000, 16000): 48,
}


class TussModel(nn.Module):
    """Task-aware unified source separation (TUSS) model presented in [1].
    Although backbone model can be any Transformer-based model, we use TF-Locoformer
    in this implementation. The encoder and decoder are based on the band-split module.

    Parameters
    ----------
    prompts: List[str]
        List of source types. The model will learn the prompt vectors for each source type,
        e.g., ["speech", "sfx", "sfxbg", "drums", "bass", "vocals", "other", "musicbg"]
    conf_cross_prompt_module: Dict
        Configuration for the cross-prompt module.
        See .yaml config files under ./configs directory for details.
    conf_cond_tse_module: Dict
        Configuration for the conditional TSE module.
    nblocks_cross_prompt_module: int
        Number of TF-Locoformer blocks in the cross-prompt module.
    nblocks_cond_tse_module: int
        Number of TF-Locoformer blocks in the conditional TSE module.
    sample_rate: int
        Sample rate of the input audio.
    stft_size: int
        STFT size of the input audio.
    eps: float
        Small constant for normalization layer.
    prompt_size: int
        Number of prompt tokens. The model will learn the prompt vectors for each prompt.
    use_sos_token: bool
        Whether to use SOS token. If True, the model will learn the SOS token vector.

    References
    ----------
    [1]: Kohei Saijo, Janek Ebbers, François G Germain, Gordon Wichern, Jonathan Le Roux,
    “Task-Aware Unified Source Separation,” Proc. ICASSP,2025.
    """

    def __init__(
        self,
        prompts: List[str],
        conf_cross_prompt_module: Dict,
        conf_cond_tse_module: Dict,
        nblocks_cross_prompt_module: int = 4,
        nblocks_cond_tse_module: int = 2,
        # band-split related
        sample_rate: int = 48000,
        stft_size: int = 2048,
        # others
        eps: float = 1.0e-5,
        prompt_size: int = 1,
        use_sos_token: bool = True,
    ):
        super().__init__()
        assert is_torch_2_0_plus, "Support only pytorch >= 2.0.0"

        #
        self.cross_prompt_module = nn.ModuleList([])
        rope_freq_first = RotaryEmbedding(
            conf_cross_prompt_module["attention_dim"] // conf_cross_prompt_module["n_heads"]
        )
        rope_time_first = RotaryEmbedding(
            conf_cross_prompt_module["attention_dim"] // conf_cross_prompt_module["n_heads"]
        )
        for _ in range(nblocks_cross_prompt_module):
            self.cross_prompt_module.append(
                TFLocoformerBlock(
                    rope_freq_first,
                    rope_time_first,
                    eps=eps,
                    **conf_cross_prompt_module,
                )
            )

        # second block
        self.cond_tse_module = nn.ModuleList([])
        rope_freq_second = RotaryEmbedding(conf_cond_tse_module["attention_dim"] // conf_cond_tse_module["n_heads"])
        rope_time_second = RotaryEmbedding(conf_cond_tse_module["attention_dim"] // conf_cond_tse_module["n_heads"])
        for _ in range(nblocks_cond_tse_module):
            self.cond_tse_module.append(
                TFLocoformerBlock(
                    rope_freq_second,
                    rope_time_second,
                    eps=eps,
                    **conf_cond_tse_module,
                )
            )

        emb_dim = conf_cross_prompt_module["emb_dim"]
        self.band_split_module = BandSplitModule(
            1,
            emb_dim,
            stft_size,
            sample_rate,
        )
        self.num_bands = len(self.band_split_module.bands)

        # prompt tokens
        self.use_sos_token = use_sos_token
        self.prompt_size = prompt_size
        self.prompts = nn.ParameterDict({})
        for prompt in prompts:
            self.prompts[prompt] = nn.Parameter(torch.randn(emb_dim, self.prompt_size, 1))  # (channel, freq)

        if self.use_sos_token:
            self.sos_token = nn.Parameter(torch.randn(emb_dim, self.prompt_size, 1))

        assert self.prompt_size == 1

    def forward(self, input: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """Forward pass of the TUSS model.

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
        n_src = len(prompts[0])

        # band-split encoder
        # normal spectrogram -> band-splitted tensor
        batch = self.band_split_module.band_split(batch)  # [B, -1, T, F]

        # Optional: concatenate SOS token
        if self.use_sos_token:
            sos_token = self.sos_token.unsqueeze(0).repeat(n_batch, 1, 1, self.num_bands)
            batch = torch.cat((sos_token, batch), dim=2)

        # concatenate learnable embeddings (prompts) to the band-splitted tensor
        # and process the tensor with the cross-prompt module
        batch = self._concatenate_prompt(batch, prompts)  # [B, -1, T+n_src, F]
        for block in self.cross_prompt_module:
            batch = block(batch)  # [B, -1, T+n_src, F]

        # split the tensor into prompt vectors and the tensor to be conditioned
        prompt_vectors, batch = (
            batch[..., : n_src * self.prompt_size, :],
            batch[..., n_src * self.prompt_size :, :],
        )
        prompt_vectors = prompt_vectors.reshape(n_batch, -1, n_src, self.prompt_size, self.num_bands).transpose(
            1, 2
        )  # (batch, n_src, channel, prompt_size, freq)

        # Optional: remove SOS token if it is used
        if self.use_sos_token:
            batch = batch[..., self.prompt_size :, :]

        # reshape the tensor to be conditioned and condition it with prompt vectors
        batch = batch.unsqueeze(1).repeat(1, n_src, 1, 1, 1)  # (batch, n_src, channel, frame, freq)
        batch = batch * prompt_vectors
        batch = batch.reshape(n_batch * n_src, -1, n_frames, self.num_bands)

        # target sound extraction
        for block in self.cond_tse_module:
            batch = block(batch, n_src=n_src)  # [B, -1, T+n_src, F]

        # band-wise decoding
        # band-splitted tensor -> normal spectrogram
        batch = self.band_split_module.bandwise_decoding(batch)  # [B, n_srcs*2, T, F]

        batch = batch.view([n_batch, n_src, 2, n_frames, n_freqs])
        batch = batch.to(torch.float32)

        # masking
        batch = torch.complex(batch[:, :, 0], batch[:, :, 1])
        batch = batch0.movedim(-1, -3) * batch
        return batch

    def _concatenate_prompt(self, input, prompt_batch):
        """Concatenate learnable prompts to the front of the encoded feature.

        Parameters
        ----------
        input: torch.Tensor, (n_batch, n_chan, n_frame, n_freq)
            A feature encoded by the band-split module.
        prompt_batch: List[List[[str]
            List of prompt strings.

        Returns
        ----------
        output: torch.Tensor, (n_batch, n_chan, n_src * prompt_size + n_frames, n_freq)
            A feature concatenated with learnable prompts.
        """
        # concatenate learnable prompts
        n_batch, n_chan, n_frame, n_freq = input.shape
        n_src = len(prompt_batch[0])
        output = input.new_zeros(n_batch, n_chan, n_frame + n_src * self.prompt_size, n_freq)
        output[..., n_src * self.prompt_size :, :] = input
        for b in range(len(prompt_batch)):
            for i, prompt in enumerate(prompt_batch[b]):
                p = self.prompts[prompt]
                p = p.repeat(1, 1, 1, n_freq)
                output[b, :, i * self.prompt_size : (i + 1) * self.prompt_size, :] = p

        return output


class BandSplitModule(nn.Module):
    def __init__(
        self,
        num_src: int,
        emb_dim: int,
        stft_size: int,
        sample_rate: int,
        emb_dim_decoder: Optional[int] = None,
    ):
        super().__init__()

        self.num_src = num_src
        num_freq_bins = stft_size // 2 + 1

        # calculate number of bins in each band
        self.bands = []
        freq_each_bin = sample_rate // 2 / num_freq_bins  # ~23.41
        for freq_range, num_bins in BAND_SPLIT.items():
            start, end = freq_range
            num_band = math.ceil((end - start) / (num_bins * freq_each_bin))
            self.bands.extend([num_bins] * num_band)

        # higher frequencies are devided into two bands
        rest = num_freq_bins - sum(self.bands)
        if sample_rate == 48000:
            self.bands.extend([rest // 4, rest // 4, rest // 4, rest // 4 + rest % 4])
        else:
            self.bands.extend([math.floor(rest / 2), math.ceil(rest / 2)])

        assert sum(self.bands) == num_freq_bins, (
            sum(self.bands),
            num_freq_bins,
            self.bands,
        )
        # print(f"Band-split module has {len(self.bands)} bands", flush=True)

        self.band_split_module = nn.ModuleList([])
        for band in self.bands:
            self.band_split_module.append(
                nn.Sequential(
                    nn.GroupNorm(1, band * 2),
                    nn.Conv1d(band * 2, emb_dim, kernel_size=1),
                )
            )

        emb_dim_decoder = emb_dim if emb_dim_decoder is None else emb_dim_decoder
        self.bandwise_decoding_module = nn.ModuleList([])
        for band in self.bands:
            self.bandwise_decoding_module.append(
                nn.Sequential(
                    nn.GroupNorm(1, emb_dim_decoder),
                    nn.Conv1d(emb_dim_decoder, emb_dim_decoder * 4, kernel_size=1),
                    nn.Tanh(),
                    nn.Conv1d(emb_dim_decoder * 4, emb_dim_decoder * 4, kernel_size=1),
                    nn.Tanh(),
                    nn.Conv1d(emb_dim_decoder * 4, band * num_src * 2 * 2, kernel_size=1),
                    nn.GLU(dim=1),
                )
            )

        self.band_idx = [0] + self.bands
        self.band_idx = list(accumulate(self.band_idx))

    def band_split(self, input):
        """Band split process

        input: torch.Tensor (n_batch, n_frame, n_freq, 2)
        """
        n_batch, n_frame = input.shape[:2]
        input = input.movedim(1, -1)

        output = []
        for b in range(len(self.bands)):
            sub_band = input[:, self.band_idx[b] : self.band_idx[b + 1]]
            output.append(self.band_split_module[b](sub_band.reshape(n_batch, -1, n_frame)))
        output = torch.stack(output, dim=-1)
        return output  # (n_batch, emb_dim, n_frame, n_bands)

    def bandwise_decoding(self, input):
        """Band-wise decoding process

        input: torch.Tensor (n_batch, emb_dim, n_frame, n_bands)
        """
        n_batch, n_frame = input.shape[0], input.shape[2]

        output = []
        for b in range(len(self.bands)):
            sub_band = self.bandwise_decoding_module[b](input[..., b])
            output.append(sub_band.reshape(n_batch, 2, self.num_src, -1, n_frame))

        return torch.cat(output, dim=-2).transpose(-1, -2)  # (n_batch, 2*num_src, n_frame, n_freq)


class TFLocoformerBlock(nn.Module):
    def __init__(
        self,
        rope_freq,
        rope_time,
        # general setup
        emb_dim=128,
        num_groups=4,
        tf_order="ft",
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        freq_ffn_config=[],
        frame_ffn_config=[],
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        assert tf_order in ["tf", "ft"], tf_order
        self.tf_order = tf_order

        self.freq_path = LocoformerBlock(
            rope_freq,
            # general setup
            emb_dim=emb_dim,
            num_groups=num_groups,
            # self-attention related
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            ffn_config=freq_ffn_config,
            dropout=dropout,
            eps=eps,
        )
        self.frame_path = LocoformerBlock(
            rope_time,
            # general setup
            emb_dim=emb_dim,
            num_groups=num_groups,
            # self-attention related
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            ffn_config=frame_ffn_config,
            dropout=dropout,
            eps=eps,
        )

    def forward(self, input, n_src=None):
        """TF-Locoformer forward.

        input: torch.Tensor
            Input tensor, (n_batch, channel, n_frame, n_freq)
        """

        if self.tf_order == "ft":
            output = self.freq_frame_process(input)
        else:
            output = self.frame_freq_process(input)

        return output

    def freq_frame_process(self, input):
        output = input.movedim(1, -1)  # (B, T, Q_old, H)
        output = self.freq_path(output)

        output = output.transpose(1, 2)  # (B, F, T, H)
        output = self.frame_path(output)
        return output.transpose(-1, 1)

    def frame_freq_process(self, input):
        # Input tensor, (n_batch, hidden, n_frame, n_freq)
        output = input.transpose(1, -1)  # (B, F, T, H)
        output = self.frame_path(output)

        output = output.transpose(1, 2)  # (B, T, F, H)
        output = self.freq_path(output)
        return output.movedim(-1, 1)


class LocoformerBlock(nn.Module):
    def __init__(
        self,
        rope,
        # general setup
        emb_dim=128,
        num_groups=4,
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        ffn_config=[],
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        # initialize FFN
        self.ffn_norm = nn.ModuleList([])
        self.ffn = nn.ModuleList([])

        assert len(ffn_config) in [1, 2], ffn_config
        self.macaron_style = len(ffn_config) == 2
        for ffn_conf in ffn_config[::-1]:
            self.ffn_norm.append(RMSGroupNorm(num_groups, emb_dim, eps=eps))
            self.ffn.append(SwiGLUConvDeconv1d(dim=emb_dim, dropout=dropout, **ffn_conf["conf"]))

        # initialize self-attention
        self.attn_norm = RMSGroupNorm(num_groups, emb_dim, eps=eps)
        self.attn = MultiHeadSelfAttention(
            emb_dim,
            attention_dim=attention_dim,
            n_heads=n_heads,
            rope=rope,
            dropout=dropout,
            flash_attention=flash_attention,
        )

    def forward(self, x):
        """Locoformer block Forward.

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either of the number of frames or freqs
        """
        B, T, F, C = x.shape

        if self.macaron_style:
            # FFN before self-attention
            input_ = x
            output = self.ffn_norm[-1](x)  # [B, T, F, C]
            output = self.ffn[-1](output)  # [B, T, F, C]
            output = output + input_
        else:
            output = x

        # Self-attention
        input_ = output
        output = self.attn_norm(output)
        output = output.contiguous().view([B * T, F, C])
        output = self.attn(output)
        output = output.contiguous().view([B, T, F, C]) + input_

        # FFN after self-attention
        input_ = output
        output = self.ffn_norm[0](output)  # [B, T, F, C]
        output = self.ffn[0](output)  # [B, T, F, C]
        output = output + input_

        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        rope=None,
        flash_attention=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        self.rope = rope
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

        if flash_attention:
            self.sdpb_backend = SDPBackend.FLASH_ATTENTION
        else:
            self.sdpb_backend = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

    def forward(self, input):
        # get query, key, and value
        query, key, value = self.get_qkv(input)

        # rotary positional encoding
        query, key = self.apply_rope(query, key)

        # self-attention
        with sdpa_kernel(self.sdpb_backend):
            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (batch, head, seq_len, -1)

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)

    def get_qkv(self, input):
        n_batch, seq_len = input.shape[:2]
        x = self.qkv(input).reshape(n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 1)  # (batch, head, seq_len, 3, -1)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value

    @torch.amp.autocast("cuda", enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key


class SwiGLUConvDeconv1d(nn.Module):
    def __init__(
        self,
        dim,
        dim_inner=None,
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()

        dim_inner = dim_inner if dim_inner is not None else dim * 4

        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)

        self.swish = nn.SiLU()
        self.deconv1d = nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x):
        """SwiGLUConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either of the number of frames or freqs
        """
        b, s1, s2, h = x.shape
        x = x.contiguous().view(b * s1, s2, h)
        x = x.transpose(-1, -2)

        # padding
        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
            + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        # conv-deconv1d
        x = self.conv1d(x)
        gate = self.swish(x[..., self.dim_inner :, :])
        x = x[..., : self.dim_inner, :] * gate
        x = self.dropout(x)
        x = self.deconv1d(x).transpose(-1, -2)

        # cut necessary part
        x = x[..., self.diff_ks : self.diff_ks + s2, :]
        return self.dropout(x).view(b, s1, s2, h)


class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim, eps=1e-8, bias=False):
        """
        Root Mean Square Group Normalization (RMSGroupNorm).
        Unlike Group Normalization in vision field, RMSGroupNorm
        is applied in each TF bins.

        Args:
            num_groups: int
                Number of groups
            dim: int
                number of dimensions
            eps: float
                Small constant to avoid zero division.
            bias: bool
                Whether to add bias term. RMSNorm does not use bias.

        """
        super().__init__()

        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // self.num_groups

        self.gamma = nn.Parameter(torch.Tensor(dim).to(torch.float32))
        nn.init.ones_(self.gamma)

        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.Tensor(dim).to(torch.float32))
            nn.init.zeros_(self.beta)
        self.eps = eps
        self.num_groups = num_groups

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, input):
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))

        # normalization
        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * self.dim_per_group ** (-1.0 / 2)
        output = input / (rms + self.eps)

        # reshape and affine transformation
        output = output.view(others + (-1,))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta

        return output
