# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import torch
import torch.nn.functional as F
from scipy import signal
from torchaudio.functional import resample as torchaudio_resample


def stft_padding(
    input_signal,
    n_fft,
    window_length,
    hop_length,
    center_pad=True,
    end_pad=True,
    pad_mode="constant",
):
    signal_length = input_signal.shape[-1]
    pad_start = 0
    pad_end = 0
    if center_pad:
        # Do center padding here instead of torch.stft, since we need to do it anyway because they don't
        # support end padding.
        pad_start = int(n_fft // 2)
        pad_end = pad_start
        signal_length = signal_length + pad_start + pad_end
    if end_pad:
        # from scipy.signal.stft
        # Pad to integer number of windowed segments
        # I.e., make signal_length = window_length + (nseg-1)*hop_length, with integer nseg
        nadd = (-(signal_length - window_length) % hop_length) % window_length
        pad_end += nadd

    # do the padding
    signal_dim = input_signal.dim()
    extended_shape = [1] * (3 - signal_dim) + list(input_signal.size())
    input_signal = F.pad(input_signal.view(extended_shape), (pad_start, pad_end), pad_mode)
    input_signal = input_signal.view(input_signal.shape[-signal_dim:])
    return input_signal


def _normalize_options(window, method_str):
    normalize_flag = False
    if method_str == "window":
        window = window / torch.sum(window)
    elif method_str == "default":
        normalize_flag = True
    return window, normalize_flag


def do_stft(
    signal,
    window_length,
    hop_length,
    fft_size=None,
    normalize="default",
    window_type="sqrt_hann",
):
    """
    Wrap torch.stft, and return transposed spectrogram for pytorch input

    :param signal: tensor of shape (n_channels, n_samples) or (n_samples)
    :param window_length: int size of stft window
    :param hop_length:  int stride of stft window
    :param fft_size: int geq to window_length, if None set to window_length
    :param normalize: string for determining how to normalize stft output.
                      "window":  divide the window by its sum.  This will give amplitudes of components that match
                                 time domain components, but small values could cause numerical issues
                      "default": default pytorch stft normalization divides by sqrt of window length
                      None:      no normalization of stft outputs
    :return: complex tensor of shape (..., n_frames, n_frequencies) or (n_frames, n_frequencies)
    """
    if fft_size is None:
        fft_size = window_length
    signal = stft_padding(signal, fft_size, window_length, hop_length)
    window = get_window(window=window_type, window_length=window_length, device=signal.device)
    window, normalize_flag = _normalize_options(window, normalize)
    result = torch.stft(
        signal,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=window_length,
        window=window,
        normalized=normalize_flag,
        center=False,
        return_complex=True,
    )
    return result.transpose(-1, -2)  # (n_batch, n_frame, n_freq)


def do_istft(
    stft,
    window_length=None,
    hop_length=None,
    fft_size=None,
    normalize="default",
    window_type="sqrt_hann",
):
    """
     Wrap torch.istft and return time domain signal

    :param stft: complex tensor of shape (n_frames, n_frequencies, n_channels) or (n_frames, n_frequencies)
    :param window_length: int size of stft window
    :param hop_length:  int stride of stft window
    :param fft_size: int geq to window_length, if None set to window_length
    :param normalize_window: divide the window by its sum.  This will give consistent values independent of fft_size,
            but small values could be more susceptible to numerical issues
    :return: tensor of shape (n_samples, n_channels) or (n_samples)
    """

    window = get_window(window=window_type, window_length=window_length, device=stft.device)
    if fft_size is None:
        fft_size = window_length
    window, normalize_flag = _normalize_options(window, normalize)
    # Must have center=True to satisfy OLA constraints

    stft = stft.transpose(-1, -2)

    if stft.ndim == 3:
        signal = torch.istft(
            stft,
            fft_size,
            hop_length=hop_length,
            win_length=window_length,
            window=window,
            center=True,
            normalized=normalize_flag,
        )
    elif stft.ndim == 4:
        signal = [
            torch.istft(
                stft[b],
                fft_size,
                hop_length=hop_length,
                win_length=window_length,
                window=window,
                center=True,
                normalized=normalize_flag,
            )
            for b in range(stft.shape[0])
        ]
        signal = torch.stack(signal, dim=0)
    else:
        raise NotImplementedError()
    return signal


def get_window(window="sqrt_hann", window_length=1024, device=None):
    if window == "sqrt_hann":
        return sqrt_hann(window_length, device=device)
    elif window == "hann":
        return torch.hann_window(window_length, periodic=True, device=device)
    elif window in ["blackman", "blackmanharris", "hamming"]:
        return torch.Tensor(signal.get_window(window, window_length), device=device)


def sqrt_hann(window_length, device=None):
    """Implement a sqrt-Hann window"""
    return torch.sqrt(torch.hann_window(window_length, periodic=True, device=device))


def get_num_fft_bins(fft_size):
    return fft_size // 2 + 1


def get_padded_stft_frames(signal_lens, window_length, hop_length, fft_size):
    """
    Compute stft frame lengths taking into account the padding used by our stft code

    :param signal_lens: torch tensor of signal waveform lengths
    :param window_length: scalar stft window length
    :param hop_length: scalar stft hop length
    :param fft_size: scalar stft fft size
    :return: torch tensor of stft frame lengths
    """
    added_padding = 2 * int(fft_size // 2)
    return torch.ceil((signal_lens + added_padding - window_length) / hop_length + 1)


def stft_3dim(input, **kwargs):
    """Apply STFT for 3-dim tensor one by one

    Parameters
    ----------
    input: torch.Tensor, (n_batch, n_samples, n_src)
        Input time-domain waveform

    Returns
    ----------
    output: torch.Tensor, (n_batch, n_frame, n_freq, n_src):
        Output STFT-domain spectrogram
    """
    assert input.ndim == 3, input.shape
    output = [
        do_stft(
            input[..., i],
            kwargs["window_length"],
            kwargs["hop_length"],
            kwargs["fft_size"],
            kwargs["normalize"],
        )
        for i in range(input.shape[-1])
    ]
    output = torch.stack(output, dim=-1).movedim(2, 0)
    return output


def istft_4dim(input, **kwargs):
    """Apply iSTFT for 4-dim tensor one by one

    Parameters
    ----------
    input: torch.Tensorm (n_batch, n_frame, n_freq, n_src)
        Input STFT-domain spectrogram

    Returns
    ----------
    output: torch.Tensor, (n_batch, n_samples, n_src)
        Output time-domain waveform
    """
    assert input.ndim == 4, input.shape
    output = [
        do_istft(
            input[i],
            kwargs["window_length"],
            kwargs["hop_length"],
            kwargs["fft_size"],
            kwargs["normalize"],
        )
        for i in range(input.shape[0])
    ]
    output = torch.stack(output)
    return output


def resample(input, sample_rate, resample_rate, method="kaiser_fast"):
    """
    Audio resampling with kaiser methods.
    Refer to https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html#performance-benchmarking.
    """
    assert method in ["kaiser_fast", "kaiser_best"], method

    if method == "kaiser_fast":
        output = torchaudio_resample(
            input,
            sample_rate,
            resample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
    else:
        output = torchaudio_resample(
            input,
            sample_rate,
            resample_rate,
            lowpass_filter_width=16,
            rolloff=0.85,
            resampling_method="sinc_interp_kaiser",
            beta=8.555504641634386,
        )
    return output
