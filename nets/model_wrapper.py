# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2025 ESPnet Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as np
import torch

from loss_functions.snr import snr_with_zeroref_loss
from utils.audio_utils import do_istft, do_stft

from .bslocoformer_stack import BSLocoformerStack
from .tuss import TussModel


def build_model(model_name, model_conf):
    if model_name == "bslocoformer_stack":
        model = BSLocoformerStack(**model_conf)
    elif model_name == "tuss":
        model = TussModel(**model_conf)
    else:
        raise ValueError(f"{model_name} is not currently supported.")

    return model


class SeparationModel(torch.nn.Module):
    def __init__(
        self,
        encoder_name,
        encoder_conf,
        decoder_name,
        decoder_conf,
        separator_name,
        separator_conf,
        css_conf,
        variance_normalization: bool = True,
    ):
        super().__init__()

        if encoder_name == "stft":
            self.encoder = partial(do_stft, **encoder_conf)
        else:
            raise NotImplementedError("Only stft encoder is supported")

        if decoder_name == "stft":
            self.decoder = partial(do_istft, **decoder_conf)
        else:
            raise NotImplementedError("Only stft decoder is supported")

        self.separator = build_model(separator_name, separator_conf)

        # settings for the css-style inference for long recordings
        # e.g., MSS or CASS
        self.css_segment_size = css_conf["segment_size"]
        self.css_shift_size = css_conf["shift_size"]
        self.css_normalize_segment_scale = css_conf["normalize_segment_scale"]
        self.css_solve_perm = css_conf["solve_perm"]
        self.fs = css_conf["sample_rate"]

        if self.css_solve_perm:
            self.css_perm_solver = partial(
                snr_with_zeroref_loss,
                snr_max=70,
                zero_ref_loss_weight=0.0,
                solve_perm=True,
                return_perm=True,
            )

        self.variance_normalization = variance_normalization

    def forward(self, mix, prompts):
        with torch.amp.autocast("cuda", enabled=False):
            if self.variance_normalization:
                std = mix.std(dim=-1, keepdims=True) + 1e-8
                mix /= std
            X = self.encoder(mix)

        Y = self.separator(X, prompts)

        with torch.amp.autocast("cuda", enabled=False):
            y = self.decoder(Y)
            if self.variance_normalization:
                y *= std.unsqueeze(1)
        return y

    def css(self, mix, prompts):
        """CSS-style separation for long recording.
        Copied from ESPnet and modified for our use.
        https://github.com/espnet/espnet/blob/master/espnet2/bin/enh_inference.py

        Assume that the input mixture is a monaural waveform.

        Args:
            mix: torch.Tensor (batch, n_samples)
                Input mixture. Must be monaural
            prompts: List[List[str]]
                List of list of strings.
                len(prompts) == n_batch and len(prompts[0]) == n_src
                e.g., [["speech", "speech"], ["sfx", "sfx"], ["musicbg", "sfxbg"]]

        Returns:
            waves: torch.Tensor (batch, n_src, n_samples)
                Separated waveforms
        """
        mix_len = mix.shape[-1]
        if mix_len > self.css_segment_size * self.fs:
            # Segment-wise speech enhancement/separation
            overlap_length = int(np.round(self.fs * (self.css_segment_size - self.css_shift_size)))
            num_segments = int(np.ceil((mix_len - overlap_length) / (self.css_shift_size * self.fs)))
            t = T = int(self.css_segment_size * self.fs)
            pad_shape = mix[:, :T].shape
            enh_waves = []

            for i in range(num_segments):
                st = int(i * self.css_shift_size * self.fs)
                en = st + T
                if en >= mix_len:
                    # en - st < T (last segment)
                    en = mix_len
                    speech_seg = mix.new_zeros(pad_shape)
                    t = en - st
                    speech_seg[:, :t] = mix[:, st:en].clone()

                else:
                    t = T
                    speech_seg = mix[:, st:en].clone()  # B x T [x C]

                if abs(speech_seg).sum() == 0.0:
                    assert i > 0, "BUG"
                    enh_waves.append(torch.zeros_like(enh_waves[-1]))
                else:
                    processed_wav = self(speech_seg, prompts)

                    if self.css_normalize_segment_scale:
                        # normalize the scale to match the input mixture scale
                        mix_energy = torch.sqrt(torch.mean(speech_seg[:, :t].pow(2), dim=1, keepdim=True))
                        enh_energy = torch.sqrt(
                            torch.mean(
                                torch.sum(processed_wav, dim=-2)[:, :t].pow(2),
                                dim=1,
                                keepdim=True,
                            )
                        )
                        processed_wav *= mix_energy / enh_energy

                    processed_wav = processed_wav[..., :T]
                    enh_waves.append(processed_wav)

            # Stitch the enhanced segments together
            waves = enh_waves[0]

            for i in range(1, num_segments):
                # permutation between separated streams in last and current segments
                if self.css_solve_perm:
                    ref_for_this_segment = waves[:, :, -overlap_length:]

                    _, perm = self.css_perm_solver(
                        enh_waves[i][:, :, :overlap_length],
                        ref_for_this_segment,
                        n_src=waves.size(1),
                    )
                    perm = perm.to(waves.device)

                    perm = perm.unsqueeze(-1).expand(-1, -1, enh_waves[i].size(-1))
                    enh_wave_tmp = torch.gather(enh_waves[i], 1, perm)
                    enh_waves[i] = enh_wave_tmp

                if i == num_segments - 1:
                    enh_waves[i][:, :, t:] = 0
                    enh_waves_res_i = enh_waves[i][:, :, overlap_length:t]
                else:
                    enh_waves_res_i = enh_waves[i][:, :, overlap_length:]

                # overlap-and-add (average over the overlapped part)
                if overlap_length > 0:
                    assert waves[:, :, -overlap_length:].shape == enh_waves[i][:, :, :overlap_length].shape

                    waves[:, :, -overlap_length:] = (
                        waves[:, :, -overlap_length:] + enh_waves[i][:, :, :overlap_length]
                    ) / 2
                # concatenate the residual parts of the later segment
                waves = torch.cat([waves, enh_waves_res_i], dim=-1)
            # ensure the stitched length is same as input
            assert waves.size(-1) == mix.size(-1), (waves.shape, mix.shape)

        else:
            # normal forward enhance for short audio
            waves = self(mix, prompts)

        return waves
