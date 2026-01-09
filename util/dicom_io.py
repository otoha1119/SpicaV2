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

# 使用するパーセンタイル（5%刻みで詳細な分布を捉える）
SIMILAR_CROP_PERCENTILES = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]

# 値域分割の閾値（低値域、中値域、高値域に分割）
SIMILAR_CROP_LOW_RANGE_THRESHOLD = 0.33   # 低値域: 0 ～ 33%
SIMILAR_CROP_HIGH_RANGE_THRESHOLD = 0.67   # 高値域: 67% ～ 100%

# 中心95%のレンジ（2.5%ile ～ 97.5%ile）
PERCENTILE_RANGE_LOW = 2.5
PERCENTILE_RANGE_HIGH = 97.5

# 最大値付近のピクセル群の判定用定数
# ピクセル数の上位何%を対象にするか（デフォルト: 5%）
SIMILAR_CROP_TOP_PIXEL_RATIO = 0.05
# 最大値の分布差の閾値（0-1の範囲、デフォルト: 0.3 = 30%）
SIMILAR_CROP_MAX_VALUE_DIFF_THRESHOLD = 0.3
# 値域の位置一致の閾値（最小値と最大値の差、0-1の範囲、デフォルト: 0.2 = 20%）
SIMILAR_CROP_RANGE_POSITION_THRESHOLD = 0.2
# 値域の重なり比率の閾値（0-1の範囲、デフォルト: 0.5 = 50%以上重なっている必要がある）
SIMILAR_CROP_RANGE_OVERLAP_THRESHOLD = 0.5

# ヒストグラムベースの類似度の閾値（KLダイバージェンス、デフォルト: 0.5）
SIMILAR_CROP_HISTOGRAM_DIFF_THRESHOLD = 0.5

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


def compute_range_segmented_stats(img: np.ndarray, low_thresh: float = 0.33, high_thresh: float = 0.67) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    画像を値域で分割して各部分の統計量を計算（血管、肺、骨を区別するため）
    
    Args:
        img: 正規化済み画像 (H, W) float32 in [0, 1]（正規化前の絶対値）
        low_thresh: 低値域の閾値（デフォルト: 0.33）
        high_thresh: 高値域の閾値（デフォルト: 0.67）
    
    Returns:
        (low_range_stats, mid_range_stats, high_range_stats): 
        各値域の[平均値, 標準偏差, パーセンタイル25, パーセンタイル50, パーセンタイル75]
    """
    img_flat = img.ravel()
    
    # 値域で分割
    low_mask = img_flat < low_thresh
    mid_mask = (img_flat >= low_thresh) & (img_flat < high_thresh)
    high_mask = img_flat >= high_thresh
    
    def compute_stats(mask):
        if np.sum(mask) == 0:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        values = img_flat[mask]
        mean_val = np.mean(values)
        std_val = np.std(values)
        p25 = np.percentile(values, 25.0)
        p50 = np.percentile(values, 50.0)
        p75 = np.percentile(values, 75.0)
        return np.array([mean_val, std_val, p25, p50, p75])
    
    low_stats = compute_stats(low_mask)
    mid_stats = compute_stats(mid_mask)
    high_stats = compute_stats(high_mask)
    
    return low_stats, mid_stats, high_stats


def compute_histogram_distance(img1: np.ndarray, img2: np.ndarray, bins: int = 50) -> float:
    """
    2つの画像のヒストグラム間の距離を計算（KLダイバージェンス風の距離）
    
    Args:
        img1: 画像1 (H, W) float32 in [0, 1]
        img2: 画像2 (H, W) float32 in [0, 1]
        bins: ヒストグラムのビン数（デフォルト: 50）
    
    Returns:
        ヒストグラム間の距離（0に近いほど類似）
    """
    # 正規化してヒストグラムを計算
    low1, high1 = compute_percentile_range(img1)
    low2, high2 = compute_percentile_range(img2)
    norm1 = normalize_for_comparison(img1, low1, high1)
    norm2 = normalize_for_comparison(img2, low2, high2)
    
    # ヒストグラムを計算
    hist1, _ = np.histogram(norm1.ravel(), bins=bins, range=(0.0, 1.0), density=True)
    hist2, _ = np.histogram(norm2.ravel(), bins=bins, range=(0.0, 1.0), density=True)
    
    # ゼロ除算を避けるため、小さな値を追加
    eps = 1e-10
    hist1 = hist1 + eps
    hist2 = hist2 + eps
    
    # 正規化
    hist1 = hist1 / (hist1.sum() + eps)
    hist2 = hist2 / (hist2.sum() + eps)
    
    # KLダイバージェンス風の距離（対称化）
    kl_12 = np.sum(hist1 * np.log((hist1 + eps) / (hist2 + eps)))
    kl_21 = np.sum(hist2 * np.log((hist2 + eps) / (hist1 + eps)))
    distance = (kl_12 + kl_21) / 2.0
    
    return float(distance)


def compute_top_pixel_stats(img: np.ndarray, top_ratio: float = 0.05) -> Tuple[float, float, float]:
    """
    正規化前の絶対値で、ピクセル数の上位N%が集まっている値域を計算
    
    Args:
        img: 正規化済み画像 (H, W) float32 in [0, 1]（正規化前の絶対値）
        top_ratio: 上位何%のピクセルを対象にするか（デフォルト: 0.05 = 5%）
    
    Returns:
        (min_value, max_value, mean_value): 上位N%のピクセルの最小値、最大値、平均値
    """
    # 正規化前の絶対値で上位N%を取得
    sorted_values = np.sort(img.ravel())
    top_count = max(1, int(len(sorted_values) * top_ratio))
    top_pixels = sorted_values[-top_count:]
    
    # 最小値、最大値、平均値
    min_value = float(np.min(top_pixels)) if len(top_pixels) > 0 else 0.0
    max_value = float(np.max(top_pixels)) if len(top_pixels) > 0 else 0.0
    mean_value = float(np.mean(top_pixels)) if len(top_pixels) > 0 else 0.0
    
    return min_value, max_value, mean_value


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
    max_value_diff_threshold: float = None,
    range_position_threshold: float = None,
    range_overlap_threshold: float = None,
    histogram_diff_threshold: float = None,
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
        max_value_diff_threshold: 最大値の分布差の閾値（Noneの場合はデフォルト値を使用）
        range_position_threshold: 値域の位置一致の閾値（Noneの場合はデフォルト値を使用）
        histogram_diff_threshold: ヒストグラム距離の閾値（Noneの場合はデフォルト値を使用）
        top_pixel_ratio: ピクセル数の上位何%を対象にするか（Noneの場合はデフォルト値を使用）
    
    Returns:
        (best_crop, retry_count): 最良のクロップとリトライ回数
    """
    if percentiles is None:
        percentiles = SIMILAR_CROP_PERCENTILES
    if median_diff_threshold is None:
        median_diff_threshold = SIMILAR_CROP_MEDIAN_DIFF_THRESHOLD
    if max_value_diff_threshold is None:
        max_value_diff_threshold = SIMILAR_CROP_MAX_VALUE_DIFF_THRESHOLD
    if range_position_threshold is None:
        range_position_threshold = SIMILAR_CROP_RANGE_POSITION_THRESHOLD
    if histogram_diff_threshold is None:
        histogram_diff_threshold = SIMILAR_CROP_HISTOGRAM_DIFF_THRESHOLD
    if top_pixel_ratio is None:
        top_pixel_ratio = SIMILAR_CROP_TOP_PIXEL_RATIO
    
    # LRクロップの上位N%の統計量を計算（正規化前の絶対値）
    ref_top_min, ref_top_max, ref_top_mean = compute_top_pixel_stats(reference_crop, top_pixel_ratio)
    
    # LRクロップの正規化パーセンタイルを計算
    ref_percentiles = compute_normalized_percentiles(reference_crop, percentiles)
    ref_median = ref_percentiles[percentiles.index(50.0)] if 50.0 in percentiles else ref_percentiles[len(percentiles)//2]
    
    # LRクロップの値域分割統計量を計算
    ref_low_stats, ref_mid_stats, ref_high_stats = compute_range_segmented_stats(reference_crop)
    
    best_crop = None
    best_distance = float('inf')
    retry_count = 0
    
    for image_retry in range(max_image_retries):
        # 候補パッチを生成
        candidates = []
        candidate_percentiles = []
        candidate_range_distances = []  # 値域分割統計量の距離を保存
        
        for _ in range(num_candidates):
            # crop_randomを使用して候補を生成
            candidate = crop_random(
                target_img, patch_size,
                require_mask=require_mask,
                min_coverage=min_coverage,
                max_tries=max_tries
            )
            
            # 候補の上位N%の統計量を計算（正規化前の絶対値）
            cand_top_min, cand_top_max, cand_top_mean = compute_top_pixel_stats(candidate, top_pixel_ratio)
            
            # 値域の位置一致をチェック（最小値と最大値の両方をチェック）
            min_diff = abs(cand_top_min - ref_top_min)
            max_diff = abs(cand_top_max - ref_top_max)
            
            # 最大値が閾値を超えたら除外（骨などの高値領域を除外）
            if max_diff > max_value_diff_threshold:
                continue  # この候補は除外
            
            # 値域の位置が大きくずれている場合は除外（最小値と最大値の両方が近い必要がある）
            if min_diff > range_position_threshold or max_diff > range_position_threshold:
                continue  # この候補は除外
            
            # 値域の重なりをチェック（上位5%の値域が十分に重なっている必要がある）
            overlap_min = max(ref_top_min, cand_top_min)
            overlap_max = min(ref_top_max, cand_top_max)
            overlap = max(0.0, overlap_max - overlap_min)
            ref_range_width = ref_top_max - ref_top_min
            cand_range_width = cand_top_max - cand_top_min
            max_range_width = max(ref_range_width, cand_range_width, 1e-10)
            overlap_ratio = overlap / max_range_width
            
            # 値域の重なりが少ない場合は除外
            if overlap_ratio < range_overlap_threshold:
                continue  # この候補は除外
            
            # 値域の中心位置と幅の差もチェック（追加の厳密性チェック）
            ref_center = (ref_top_min + ref_top_max) / 2.0
            cand_center = (cand_top_min + cand_top_max) / 2.0
            center_diff = abs(cand_center - ref_center)
            width_diff = abs(cand_range_width - ref_range_width)
            
            # 中心位置または幅の差が大きすぎる場合は除外
            if center_diff > range_position_threshold or width_diff > range_position_threshold:
                continue  # この候補は除外
            
            # ヒストグラム距離を計算
            hist_distance = compute_histogram_distance(reference_crop, candidate)
            if hist_distance > histogram_diff_threshold:
                continue  # この候補は除外
            
            candidates.append(candidate)
            
            # 候補の正規化パーセンタイルを計算
            cand_percentiles = compute_normalized_percentiles(candidate, percentiles)
            candidate_percentiles.append(cand_percentiles)
            
            # 候補の値域分割統計量を計算
            cand_low_stats, cand_mid_stats, cand_high_stats = compute_range_segmented_stats(candidate)
            
            # 値域分割統計量の距離を計算（追加の類似度指標）
            low_dist = np.linalg.norm(cand_low_stats - ref_low_stats)
            mid_dist = np.linalg.norm(cand_mid_stats - ref_mid_stats)
            high_dist = np.linalg.norm(cand_high_stats - ref_high_stats)
            # 値域分割統計量の総合距離（重み付き平均）
            range_distance = (low_dist + mid_dist * 2.0 + high_dist) / 4.0  # 中値域を重視
            candidate_range_distances.append(range_distance)
        
        # 各候補とLRクロップの距離を計算（パーセンタイル距離 + 値域分割統計量距離）
        candidate_percentiles = np.array(candidate_percentiles)
        percentile_distances = np.linalg.norm(candidate_percentiles - ref_percentiles, axis=1)
        range_distances = np.array(candidate_range_distances)
        
        # 正規化して組み合わせ（パーセンタイル距離を重視）
        if len(percentile_distances) > 0:
            percentile_dist_norm = percentile_distances / (percentile_distances.max() + 1e-10)
            range_dist_norm = range_distances / (range_distances.max() + 1e-10) if range_distances.max() > 0 else range_distances
            # パーセンタイル距離70%、値域分割距離30%の重みで組み合わせ
            combined_distances = 0.7 * percentile_dist_norm + 0.3 * range_dist_norm
        else:
            combined_distances = percentile_distances
        
        best_idx = np.argmin(combined_distances)
        
        # 最良候補の中央値を確認
        best_cand_percentiles = candidate_percentiles[best_idx]
        best_cand_median = best_cand_percentiles[percentiles.index(50.0)] if 50.0 in percentiles else best_cand_percentiles[len(percentiles)//2]
        median_diff = abs(best_cand_median - ref_median)
        
        # より良い候補が見つかった場合、または閾値以下なら採用
        if combined_distances[best_idx] < best_distance:
            best_distance = combined_distances[best_idx]
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