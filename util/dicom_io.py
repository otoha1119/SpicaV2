from __future__ import annotations
import os, time
from typing import List, Optional, Tuple
import numpy as np

try:
    import pydicom
    from pydicom.uid import generate_uid
except Exception as e:
    pydicom = None

# ====== 類似クロップ用の定数 ======
# 中央値の差の閾値（0-1の範囲、15% = 0.15）
# この値以上差がある場合はHR画像を選び直す
SIMILAR_CROP_MEDIAN_DIFF_THRESHOLD = 0.15

# 使用するパーセンタイル（25%, 50%, 75%）
SIMILAR_CROP_PERCENTILES = [25.0, 50.0, 75.0]

# 中心95%のレンジ（2.5%ile ～ 97.5%ile）
PERCENTILE_RANGE_LOW = 2.5
PERCENTILE_RANGE_HIGH = 97.5

# 最大値付近のピクセル群の判定用定数
# ピクセル数の上位何%を対象にするか（デフォルト: 5%）
SIMILAR_CROP_TOP_PIXEL_RATIO = 0.05
# 最大値付近の分布差の閾値（0-1の範囲、デフォルト: 0.3 = 30%）
SIMILAR_CROP_MAX_DIFF_THRESHOLD = 0.3

def require_pydicom():
    if pydicom is None:
        raise ImportError("pydicom is required but not installed. Please `pip install pydicom`.")

def list_dicom_files_recursive(root_dir: str) -> List[str]:
    files = []
    for d, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.lower().endswith(".dcm") or fn.lower().endswith(".dicom"):
                files.append(os.path.join(d, fn))
    return sorted(files)

def sort_series_paths(paths: List[str]) -> List[str]:
    require_pydicom()
    def key_fn(p):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, specific_tags=["ImagePositionPatient","InstanceNumber"])
            if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient and len(ds.ImagePositionPatient)==3:
                return float(ds.ImagePositionPatient[2])
            if hasattr(ds, "InstanceNumber"):
                return int(ds.InstanceNumber)
        except Exception:
            pass
        return p
    return sorted(paths, key=key_fn)

def read_normalized_pixels(path: str) -> np.ndarray:
    require_pydicom()
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.int32)
    norm = (arr + 2048.0) / 6143.0
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
    return norm

def denormalize_to_int16(norm: np.ndarray) -> np.ndarray:
    return np.rint(norm * 6143.0 - 2048.0).astype(np.int16)

def compute_body_mask(norm_img: np.ndarray, thresh_norm: float = 0.1, min_area: int = 64) -> np.ndarray:
    mask = (norm_img > thresh_norm).astype(np.uint8)
    try:
        from scipy import ndimage as ndi
        labeled, n = ndi.label(mask)
        if n <= 1:
            return mask
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest = counts.argmax()
        mask = (labeled == largest).astype(np.uint8)
        mask = ndi.binary_closing(mask, iterations=1).astype(np.uint8)
        return mask
    except Exception:
        return mask

def crop_random(norm_img: np.ndarray, patch_size: int, require_mask: Optional[np.ndarray]=None, min_coverage: float=0.0, max_tries: int=32) -> np.ndarray:
    H, W = norm_img.shape[:2]
    ps = patch_size
    if H < ps or W < ps:
        pad_h = max(0, ps - H)
        pad_w = max(0, ps - W)
        norm_img = np.pad(norm_img, ((pad_h//2, pad_h - pad_h//2),(pad_w//2, pad_w - pad_w//2)), mode="reflect")
        if require_mask is not None:
            require_mask = np.pad(require_mask, ((pad_h//2, pad_h - pad_h//2),(pad_w//2, pad_w - pad_w//2)), mode="constant")
        H, W = norm_img.shape[:2]
    for _ in range(max_tries):
        y = np.random.randint(0, H - ps + 1)
        x = np.random.randint(0, W - ps + 1)
        crop = norm_img[y:y+ps, x:x+ps]
        if require_mask is None or min_coverage <= 0.0:
            return crop
        m = require_mask[y:y+ps, x:x+ps]
        if (m>0).mean() >= min_coverage:
            return crop
    y = (H - ps)//2; x = (W - ps)//2
    return norm_img[y:y+ps, x:x+ps]

def compute_percentile_range(img: np.ndarray) -> Tuple[float, float]:
    """
    画像の中心95%の値レンジ（2.5%ile～97.5%ile）を計算
    
    Args:
        img: 正規化済み画像 (H, W) float32 in [0, 1]
    
    Returns:
        (low, high): 2.5%ileと97.5%ileの値
    """
    low = np.percentile(img, PERCENTILE_RANGE_LOW)
    high = np.percentile(img, PERCENTILE_RANGE_HIGH)
    return float(low), float(high)


def normalize_for_comparison(img: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    画像を中心95%レンジで正規化（統計量比較用）
    
    Args:
        img: 正規化済み画像 (H, W) float32 in [0, 1]
        low: 2.5%ileの値
        high: 97.5%ileの値
    
    Returns:
        正規化された画像（値域は[0, 1]にマッピング、範囲外はクリップ）
    """
    if high <= low:
        # レンジが0または負の場合、そのまま返す
        return img
    normalized = (img - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def compute_normalized_percentiles(img: np.ndarray, percentiles: List[float]) -> np.ndarray:
    """
    画像を中心95%レンジで正規化してからパーセンタイルを計算
    
    Args:
        img: 正規化済み画像 (H, W) float32 in [0, 1]
        percentiles: 計算するパーセンタイル値のリスト（例: [25.0, 50.0, 75.0]）
    
    Returns:
        パーセンタイル値の配列 (len(percentiles),)
    """
    low, high = compute_percentile_range(img)
    normalized = normalize_for_comparison(img, low, high)
    return np.array([np.percentile(normalized, p) for p in percentiles])


def compute_top_pixel_stats(img: np.ndarray, top_ratio: float = 0.05) -> Tuple[float, float]:
    """
    正規化後の画像で、ピクセル数の上位N%の統計量を計算
    
    Args:
        img: 正規化済み画像 (H, W) float32 in [0, 1]
        top_ratio: 上位何%のピクセルを対象にするか（デフォルト: 0.05 = 5%）
    
    Returns:
        (mean_value, max_value): 上位N%のピクセルの平均値と最大値
    """
    low, high = compute_percentile_range(img)
    normalized = normalize_for_comparison(img, low, high)
    
    # ピクセル数の上位N%を取得
    sorted_values = np.sort(normalized.ravel())
    top_count = max(1, int(len(sorted_values) * top_ratio))
    top_pixels = sorted_values[-top_count:]
    
    # 平均値と最大値
    mean_value = np.mean(top_pixels) if len(top_pixels) > 0 else 0.0
    max_value = np.max(top_pixels) if len(top_pixels) > 0 else 0.0
    
    return float(mean_value), float(max_value)


def crop_similar_with_retry(
    target_img: np.ndarray,
    reference_crop: np.ndarray,
    patch_size: int,
    num_candidates: int = 32,
    percentiles: List[float] = None,
    require_mask: Optional[np.ndarray] = None,
    min_coverage: float = 0.0,
    max_tries: int = 32,
    max_image_retries: int = 3,
    median_diff_threshold: float = None,
    max_diff_threshold: float = None,
    top_pixel_ratio: float = None
) -> Tuple[np.ndarray, int]:
    """
    LRクロップに類似したHRクロップを見つける（HR画像選び直し対応）
    
    Args:
        target_img: HR画像全体 (H, W) float32 in [0, 1]
        reference_crop: LRクロップ (patch_size, patch_size) float32 in [0, 1]
        patch_size: パッチサイズ
        num_candidates: HR画像から生成する候補パッチ数
        percentiles: 使用するパーセンタイル値のリスト（Noneの場合はデフォルト値を使用）
        require_mask: ボディマスク（オプション）
        min_coverage: マスクの最小カバレッジ
        max_tries: 候補パッチ生成の最大試行回数
        max_image_retries: HR画像選び直しの最大回数
        median_diff_threshold: 中央値差の閾値（Noneの場合はデフォルト値を使用）
        max_diff_threshold: 最大値付近の分布差の閾値（Noneの場合はデフォルト値を使用）
        top_pixel_ratio: ピクセル数の上位何%を対象にするか（Noneの場合はデフォルト値を使用）
    
    Returns:
        (best_crop, retry_count): 最良のクロップとリトライ回数
    """
    if percentiles is None:
        percentiles = SIMILAR_CROP_PERCENTILES
    if median_diff_threshold is None:
        median_diff_threshold = SIMILAR_CROP_MEDIAN_DIFF_THRESHOLD
    if max_diff_threshold is None:
        max_diff_threshold = SIMILAR_CROP_MAX_DIFF_THRESHOLD
    if top_pixel_ratio is None:
        top_pixel_ratio = SIMILAR_CROP_TOP_PIXEL_RATIO
    
    # LRクロップの上位N%の統計量を計算
    ref_top_mean, ref_top_max = compute_top_pixel_stats(reference_crop, top_pixel_ratio)
    
    # LRクロップの正規化パーセンタイルを計算
    ref_percentiles = compute_normalized_percentiles(reference_crop, percentiles)
    ref_median = ref_percentiles[percentiles.index(50.0)] if 50.0 in percentiles else ref_percentiles[len(percentiles)//2]
    
    best_crop = None
    best_distance = float('inf')
    retry_count = 0
    
    for image_retry in range(max_image_retries):
        # 候補パッチを生成
        candidates = []
        candidate_percentiles = []
        
        for _ in range(num_candidates):
            # crop_randomを使用して候補を生成
            candidate = crop_random(
                target_img, patch_size,
                require_mask=require_mask,
                min_coverage=min_coverage,
                max_tries=max_tries
            )
            
            # 候補の上位N%の統計量を計算して、最大値付近の分布が大きく異なる場合は除外
            cand_top_mean, cand_top_max = compute_top_pixel_stats(candidate, top_pixel_ratio)
            mean_diff = abs(cand_top_mean - ref_top_mean)
            max_diff = abs(cand_top_max - ref_top_max)
            
            # 平均値または最大値の差が閾値を超えたら除外
            if mean_diff > max_diff_threshold or max_diff > max_diff_threshold:
                continue  # この候補は除外
            
            candidates.append(candidate)
            
            # 候補の正規化パーセンタイルを計算
            cand_percentiles = compute_normalized_percentiles(candidate, percentiles)
            candidate_percentiles.append(cand_percentiles)
        
        # 各候補とLRクロップのL2距離を計算
        candidate_percentiles = np.array(candidate_percentiles)
        distances = np.linalg.norm(candidate_percentiles - ref_percentiles, axis=1)
        best_idx = np.argmin(distances)
        
        # 最良候補の中央値を確認
        best_cand_percentiles = candidate_percentiles[best_idx]
        best_cand_median = best_cand_percentiles[percentiles.index(50.0)] if 50.0 in percentiles else best_cand_percentiles[len(percentiles)//2]
        median_diff = abs(best_cand_median - ref_median)
        
        # より良い候補が見つかった場合、または閾値以下なら採用
        if distances[best_idx] < best_distance:
            best_distance = distances[best_idx]
            best_crop = candidates[best_idx]
            retry_count = image_retry
        
        # 中央値差が閾値以下なら採用して終了
        if median_diff <= median_diff_threshold:
            break
    
    # 最良候補を返す（リトライ上限に達した場合も含む）
    if best_crop is None:
        # フォールバック: 中央からクロップ
        H, W = target_img.shape[:2]
        ps = patch_size
        if H < ps or W < ps:
            pad_h = max(0, ps - H)
            pad_w = max(0, ps - W)
            target_img = np.pad(target_img, ((pad_h//2, pad_h - pad_h//2),(pad_w//2, pad_w - pad_w//2)), mode="reflect")
            H, W = target_img.shape[:2]
        y = (H - ps) // 2
        x = (W - ps) // 2
        best_crop = target_img[y:y+ps, x:x+ps]
    
    return best_crop, retry_count


def save_dicom_like(reference_path: str, output_path: str, norm_img: np.ndarray, halves_pixel_spacing: bool=True):
    require_pydicom()
    ref = pydicom.dcmread(reference_path)
    out = ref.copy()
    img_int16 = denormalize_to_int16(norm_img.astype(np.float32))
    H, W = img_int16.shape
    out.Rows = int(H); out.Columns = int(W)
    out.SamplesPerPixel = 1
    out.PhotometricInterpretation = "MONOCHROME2"
    try:
        if hasattr(out,"PixelSpacing") and out.PixelSpacing and len(out.PixelSpacing)==2 and halves_pixel_spacing:
            out.PixelSpacing = [str(float(out.PixelSpacing[0])/2.0), str(float(out.PixelSpacing[1])/2.0)]
    except Exception:
        pass
    out.BitsAllocated = 16; out.BitsStored = 16; out.HighBit = 15
    out.PixelRepresentation = 1
    out.SmallestImagePixelValue = int(img_int16.min())
    out.LargestImagePixelValue  = int(img_int16.max())
    out.RescaleSlope = 1; out.RescaleIntercept = 0
    from pydicom.uid import CTImageStorage
    try:
        if not hasattr(out,"SOPClassUID") or not out.SOPClassUID:
            out.SOPClassUID = CTImageStorage
    except Exception:
        pass
    out.SeriesInstanceUID = generate_uid()
    out.SOPInstanceUID    = generate_uid()
    out.InstanceCreationDate = time.strftime("%Y%m%d")
    out.InstanceCreationTime = time.strftime("%H%M%S")
    out.PixelData = img_int16.tobytes()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.save_as(output_path, write_like_original=False)