"""
Core MTF computation routines (slanted-edge).
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

# skimage の gaussian が無い環境でも動くようにフォールバック
try:
    from skimage.filters import gaussian  # type: ignore
except Exception:
    gaussian = None  # type: ignore


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def _find_threshold_freq(freq: np.ndarray, mtf: np.ndarray, threshold: float) -> float:
    for i in range(1, len(mtf)):
        if (mtf[i - 1] >= threshold and mtf[i] <= threshold) or (mtf[i - 1] <= threshold and mtf[i] >= threshold):
            f0, f1 = freq[i - 1], freq[i]
            m0, m1 = mtf[i - 1], mtf[i]
            if abs(m1 - m0) < 1e-12:
                return float(f0)
            ratio = (threshold - m0) / (m1 - m0)
            return float(f0 + ratio * (f1 - f0))
    return float(freq[-1]) if len(freq) else 0.0


def compute_mtf_for_roi(
    roi: np.ndarray,
    pixel_spacing: Tuple[float, float],
    oversample_factor: int = 4,
    zero_pad_factor: int = 4,
    use_cuda: bool = False,
    row_fraction: float = 0.6,
    reduce_mode: str = "median",   # 既定: median で安定化
    trim_alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    回転済み ROI（エッジが縦）から MTF を算出。
    """
    if roi.ndim != 2:
        raise ValueError("ROI must be 2D")
    h, w = roi.shape

    # --- ROI中央帯のみ使用 & 堅牢な縮約で ESF 作成 ---
    frac = float(np.clip(row_fraction, 1e-3, 1.0))
    band = int(max(1, round(h * frac)))
    y0 = (h - band) // 2
    slab = roi[y0:y0 + band, :]

    if reduce_mode == "median":
        esf = np.median(slab, axis=0)
    elif reduce_mode == "trimmed":
        alpha = float(np.clip(trim_alpha, 0.0, 0.49))
        k_low = int(np.floor(alpha * slab.shape[0]))
        k_high = slab.shape[0] - int(np.floor(alpha * slab.shape[0]))
        sort_slab = np.sort(slab, axis=0)
        core = sort_slab[k_low:k_high, :]
        esf = np.mean(core, axis=0)
    else:
        esf = np.mean(slab, axis=0)
    esf = esf.astype(np.float64)

    # --- ESF のオーバーサンプル ---
    x_original = np.arange(len(esf))
    x_interp = np.linspace(0, len(esf) - 1, max(2, len(esf) * oversample_factor), endpoint=True)
    esf_oversampled = np.interp(x_interp, x_original, esf)

    # --- 平滑化（skimage 無ければ移動平均） ---
    if gaussian is not None:
        esf_smooth = gaussian(esf_oversampled, sigma=1.0, mode="nearest")
    else:
        k = max(5, int(oversample_factor * 3))
        ker = np.ones(k, dtype=float) / k
        esf_smooth = np.convolve(esf_oversampled, ker, mode="same")

    # --- LSF → Hann → rFFT ---
    lsf = np.diff(esf_smooth)
    if len(lsf) < 2:
        raise ValueError("LSF too short")
    window = np.hanning(len(lsf))
    lsf_windowed = lsf * window

    n_fft = next_power_of_two(int(len(lsf_windowed) * zero_pad_factor))
    lsf_padded = np.pad(lsf_windowed, (0, n_fft - len(lsf_windowed)), mode="constant")
    mtf_complex = np.fft.rfft(lsf_padded)
    mtf = np.abs(mtf_complex)

    # 周波数軸（cycles/mm）
    col_spacing = float(pixel_spacing[1])
    sample_spacing = col_spacing / float(oversample_factor)
    freq = np.fft.rfftfreq(n_fft, d=sample_spacing)

    # DC 正規化
    if mtf.size > 0 and mtf[0] != 0.0:
        mtf = mtf / mtf[0]

    mtf50 = _find_threshold_freq(freq, mtf, 0.5)
    mtf10 = _find_threshold_freq(freq, mtf, 0.1)
    return freq, mtf, mtf50, mtf10


def compute_auc(freq: np.ndarray, mtf: np.ndarray, f_max: float) -> float:
    if f_max <= 0 or len(freq) == 0:
        return 0.0
    mask = freq <= f_max
    if not np.any(mask):
        return 0.0
    return float(np.trapz(mtf[mask], freq[mask]))
