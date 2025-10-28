"""
Core MTF computation routines.

This module contains functions to convert a rotated ROI into a
modulation transfer function (MTF) curve using the slanted‑edge
approach.  It relies on 1D oversampling of the edge spread function
(ESF), smoothing, differentiation to obtain the line spread function
(LSF), application of a Hann window, and FFT.  The resulting MTF is
normalised at zero frequency.  Additional helper functions compute
standard metrics such as MTF50, MTF10 and the area under the MTF
curve up to a specified frequency.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from skimage.filters import gaussian  # type: ignore

from .utils import next_power_of_two


def _find_threshold_freq(freq: np.ndarray, mtf: np.ndarray, threshold: float) -> float:
    """
    Find the frequency at which the MTF crosses a given threshold.

    Linear interpolation is used between the two bins around the threshold.
    If the MTF never falls below the threshold, the last frequency is returned.
    """
    if len(freq) != len(mtf):
        raise ValueError("freq and mtf lengths must match")
    for i in range(1, len(mtf)):
        if (mtf[i - 1] >= threshold and mtf[i] <= threshold) or (mtf[i - 1] <= threshold and mtf[i] >= threshold):
            # Linear interpolation
            f0, f1 = freq[i - 1], freq[i]
            m0, m1 = mtf[i - 1], mtf[i]
            if abs(m1 - m0) < 1e-12:
                return float(f0)
            ratio = (threshold - m0) / (m1 - m0)
            return float(f0 + ratio * (f1 - f0))
    # If threshold never crossed, return last frequency
    return float(freq[-1])


def compute_mtf_for_roi(
    roi: np.ndarray,
    pixel_spacing: Tuple[float, float],
    oversample_factor: int = 4,
    zero_pad_factor: int = 4,
    use_cuda: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute the MTF curve for a single ROI.

    Parameters
    ----------
    roi : np.ndarray
        Rotated ROI of shape (height, width) oriented so that the edge is vertical.
    pixel_spacing : tuple (row_spacing, col_spacing) in mm/pixel
        Pixel spacing for the slice from which this ROI originates.  Only the
        along‑row spacing (normal to the edge) is used here.  The MTF is
        expressed per millimetre using this spacing.
    oversample_factor : int, optional
        Factor by which to oversample the ESF.  Higher values produce
        smoother MTFs at the cost of computation time.
    zero_pad_factor : int, optional
        Multiplier controlling zero padding applied prior to the FFT.  The
        length of the LSF after windowing is multiplied by this factor and
        rounded up to the next power of two.  A larger value gives finer
        frequency resolution.
    use_cuda : bool, optional
        If True and a CUDA‑enabled PyTorch installation is available, the
        FFT will be computed on the GPU.  Otherwise NumPy is used.

    Returns
    -------
    freq : np.ndarray
        Array of spatial frequencies (cycles/mm) corresponding to the MTF samples.
    mtf : np.ndarray
        Normalised MTF values (0–1).
    mtf50 : float
        Frequency (cycles/mm) at which the MTF drops to 0.5.
    mtf10 : float
        Frequency (cycles/mm) at which the MTF drops to 0.1.
    """
    if roi.ndim != 2:
        raise ValueError("ROI must be 2D")
    # Compute ESF: mean across the long dimension (rows), producing 1D array across the normal direction
    esf = np.mean(roi, axis=0).astype(np.float64)
    # Oversample the ESF by interpolation
    w = len(esf)
    x_original = np.arange(w)
    x_interp = np.linspace(0, w - 1, w * oversample_factor, endpoint=True)
    esf_oversampled = np.interp(x_interp, x_original, esf)
    # Smooth ESF with a small Gaussian kernel to reduce noise (Savitzky–Golay like)
    esf_smooth = gaussian(esf_oversampled, sigma=1.0, mode="nearest")
    # Compute LSF (first derivative)
    lsf = np.diff(esf_smooth)
    # Apply Hann window to reduce spectral leakage
    if len(lsf) < 2:
        raise ValueError("LSF length too short")
    window = np.hanning(len(lsf))
    lsf_windowed = lsf * window
    # Zero‑pad LSF to increase frequency resolution
    n_fft = next_power_of_two(int(len(lsf_windowed) * zero_pad_factor))
    # Compute FFT either on CPU or GPU
    mtf: np.ndarray
    try:
        if use_cuda:
            import torch  # type: ignore

            if torch.cuda.is_available():
                device = torch.device("cuda")
                lsf_tensor = torch.from_numpy(lsf_windowed.astype(np.float32)).to(device)
                pad_len = n_fft - lsf_tensor.shape[0]
                lsf_tensor = torch.cat([lsf_tensor, torch.zeros(pad_len, device=device)])
                mtf_complex = torch.fft.rfft(lsf_tensor)
                mtf = torch.abs(mtf_complex).cpu().numpy()
            else:
                raise RuntimeError("CUDA requested but not available")
        else:
            raise RuntimeError("CPU path")
    except Exception:
        # Fallback to NumPy CPU implementation
        lsf_padded = np.pad(lsf_windowed, (0, n_fft - len(lsf_windowed)), mode="constant")
        mtf_complex = np.fft.rfft(lsf_padded)
        mtf = np.abs(mtf_complex)
    # Frequency axis in cycles/mm.  Sample spacing (mm) = pixel_spacing[1] / oversample_factor
    # Pixel spacing: (row_spacing, col_spacing).  The normal direction corresponds to columns.
    col_spacing = float(pixel_spacing[1])
    sample_spacing = col_spacing / float(oversample_factor)
    freq = np.fft.rfftfreq(n_fft, d=sample_spacing)
    # Normalise MTF at DC (index 0)
    if mtf[0] != 0.0:
        mtf = mtf / mtf[0]
    # Interpolate MTF50 and MTF10
    mtf50 = _find_threshold_freq(freq, mtf, 0.5)
    mtf10 = _find_threshold_freq(freq, mtf, 0.1)
    return freq, mtf, mtf50, mtf10


def compute_auc(
    freq: np.ndarray,
    mtf: np.ndarray,
    f_max: float,
) -> float:
    """
    Compute the area under the MTF curve up to the specified maximum frequency.

    Parameters
    ----------
    freq : np.ndarray
        Frequencies corresponding to the MTF values (cycles/mm).
    mtf : np.ndarray
        MTF values (normalised between 0 and 1).
    f_max : float
        Upper limit of integration (cycles/mm).  Values beyond this limit are ignored.

    Returns
    -------
    float
        The area under the MTF curve up to f_max.
    """
    # Select portion of the MTF within range
    if f_max <= 0 or len(freq) == 0:
        return 0.0
    mask = freq <= f_max
    if not np.any(mask):
        return 0.0
    freq_sub = freq[mask]
    mtf_sub = mtf[mask]
    # Trapezoidal integration
    auc = np.trapz(mtf_sub, freq_sub)
    return float(auc)