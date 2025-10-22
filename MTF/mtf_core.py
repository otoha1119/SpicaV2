"""
Core MTF computation routines.

Slanted-edge pipeline:
ESF (row-averaged along normal) -> oversample (x4) -> Gaussian smoothing
-> diff -> Hann window -> zero-pad -> rFFT -> DC normalization
Also returns MTF50 / MTF10 and provides AUC helper.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _interp_1d(y: np.ndarray, factor: int) -> np.ndarray:
    """Linear oversampling by 'factor'."""
    w = len(y)
    if w <= 1 or factor <= 1:
        return y.astype(np.float64, copy=True)
    x0 = np.arange(w, dtype=np.float64)
    x1 = np.linspace(0.0, float(w - 1), int(w * factor), endpoint=True)
    return np.interp(x1, x0, y.astype(np.float64))


def _gaussian_1d(y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(y.astype(np.float64), sigma=sigma, mode="nearest")
    except Exception:
        # simple fallback: discrete Gaussian kernel
        if sigma <= 0:
            return y.astype(np.float64, copy=True)
        ksize = int(np.ceil(sigma * 6)) | 1  # odd
        half = ksize // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        kern = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
        kern /= np.sum(kern)
        return np.convolve(y.astype(np.float64), kern, mode="same")


def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float64)
    return np.hanning(n)


def _linear_crossing(x: np.ndarray, y: np.ndarray, thr: float) -> float:
    """Find x where y crosses 'thr' using linear interpolation."""
    for i in range(1, len(y)):
        y0, y1 = y[i - 1], y[i]
        if (y0 >= thr and y1 <= thr) or (y0 <= thr and y1 >= thr):
            x0, x1 = x[i - 1], x[i]
            if abs(y1 - y0) < 1e-12:
                return float(x0)
            t = (thr - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))
    return float("nan")


def compute_esf_lsf_mtf(
    roi: np.ndarray,
    pixel_spacing: Tuple[float, float],
    oversample_factor: int = 4,
    zero_pad_factor: int = 4,
    use_cuda: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Compute ESF/LSF/MTF for a single rotated ROI (edge vertical; normal along columns).

    Parameters
    ----------
    roi : (H, W) array
        Rotated ROI, where the edge normal is the column direction.
    pixel_spacing : (row_spacing_mm, col_spacing_mm)
        DICOM PixelSpacing (mm). Frequency axis uses col_spacing.
    oversample_factor : int
        ESF oversampling factor.
    zero_pad_factor : int
        Multiplier for zero-padding length before FFT (rounded to next pow2).
    use_cuda : bool
        If True and torch with CUDA is available, FFT runs on GPU.

    Returns
    -------
    esf_oversampled : (N,) ESF after oversampling and smoothing
    lsf_windowed    : (N-1,) LSF after diff and Hann window
    freq            : (K,) frequency axis [cycles/mm]
    mtf             : (K,) normalized MTF (DC = 1)
    mtf50           : scalar, frequency at 50% MTF
    mtf10           : scalar, frequency at 10% MTF
    """
    if roi.ndim != 2:
        raise ValueError("ROI must be 2D")

    # ESF: average along rows -> profile along columns
    esf = np.mean(roi.astype(np.float64), axis=0)
    # Oversample + smooth
    esf_oversampled = _interp_1d(esf, oversample_factor)
    esf_smooth = _gaussian_1d(esf_oversampled, sigma=1.0)
    # LSF: first difference + Hann window
    lsf = np.diff(esf_smooth)
    if len(lsf) < 2:
        raise RuntimeError("LSF length too short. Check ROI size / rotation.")
    lsf_windowed = lsf * _hann(len(lsf))
    # Zero-pad & rFFT
    n_fft = _next_power_of_two(int(len(lsf_windowed) * max(1, zero_pad_factor)))
    try:
        if use_cuda:
            import torch
            if torch.cuda.is_available():
                dev = torch.device("cuda")
                t = torch.from_numpy(lsf_windowed.astype(np.float32)).to(dev)
                if n_fft > t.numel():
                    t = torch.cat([t, torch.zeros(n_fft - t.numel(), device=dev)])
                mtf_c = torch.fft.rfft(t)
                mtf = torch.abs(mtf_c).detach().cpu().numpy().astype(np.float64)
            else:
                raise RuntimeError("CUDA requested but not available")
        else:
            raise RuntimeError("CPU path")
    except Exception:
        lsf_padded = np.pad(lsf_windowed, (0, max(0, n_fft - len(lsf_windowed))), mode="constant")
        mtf_c = np.fft.rfft(lsf_padded)
        mtf = np.abs(mtf_c).astype(np.float64)

    # Frequency axis (cycles/mm). Sample pitch along normal:
    col_spacing = float(pixel_spacing[1])  # mm/px
    sample_spacing = col_spacing / float(max(1, oversample_factor))  # mm per oversampled sample
    freq = np.fft.rfftfreq(n_fft, d=sample_spacing)

    # DC normalization
    if mtf[0] != 0.0:
        mtf = mtf / mtf[0]

    # Threshold metrics
    mtf50 = _linear_crossing(freq, mtf, 0.5)
    mtf10 = _linear_crossing(freq, mtf, 0.1)

    return esf_oversampled, lsf_windowed, freq, mtf, float(mtf50), float(mtf10)


def compute_mtf_for_roi(
    roi: np.ndarray,
    pixel_spacing: Tuple[float, float],
    oversample_factor: int = 4,
    zero_pad_factor: int = 4,
    use_cuda: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Backward-compatible convenience wrapper:
    returns (freq, mtf, mtf50, mtf10)
    """
    _, _, freq, mtf, mtf50, mtf10 = compute_esf_lsf_mtf(
        roi=roi,
        pixel_spacing=pixel_spacing,
        oversample_factor=oversample_factor,
        zero_pad_factor=zero_pad_factor,
        use_cuda=use_cuda,
    )
    return freq, mtf, float(mtf50), float(mtf10)


def compute_auc(freq: np.ndarray, mtf: np.ndarray, f_max: float) -> float:
    """
    Area under the MTF curve up to f_max (cycles/mm).
    Uses trapezoidal integration.
    """
    if f_max <= 0 or len(freq) == 0:
        return 0.0
    mask = freq <= f_max
    if not np.any(mask):
        return 0.0
    return float(np.trapz(mtf[mask], freq[mask]))
