from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2

@dataclass
class ROI:
    series_name: str
    slice_index: int
    slice_filename: str
    roi_image: np.ndarray
    orientation_deg: float
    delta_hu: float
    group: int

class ROISelector:
    def __init__(
        self,
        roi_width: int = 40,
        roi_height: int = 100,
        angle_min: float = 6.0,
        angle_max: float = 12.0,
        delta_hu_threshold: float = 200.0,
        edge_vertical_tol_deg: float = 2.5,
        min_vertical_span_ratio: float = 0.92,
        require_single_edge: bool = True,
        center_tolerance_ratio: float = 0.08,
    ):
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.delta_hu_threshold = delta_hu_threshold
        self.edge_vertical_tol_deg = edge_vertical_tol_deg
        self.min_vertical_span_ratio = min_vertical_span_ratio
        self.require_single_edge = require_single_edge
        self.center_tolerance_ratio = center_tolerance_ratio

    def _extract_roi_from_candidate(self, img: np.ndarray, rc: Tuple[int, int], angle_deg: float) -> np.ndarray | None:
        r, c = rc
        h, w = img.shape
        # 回転して「エッジが縦」になるように補正
        M = cv2.getRotationMatrix2D((c, r), -angle_deg, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        # 回転後の中心近傍を切り出す（正味は候補画素周り）
        x1 = int(c - self.roi_width // 2)
        y1 = int(r - self.roi_height // 2)
        x2 = x1 + self.roi_width
        y2 = y1 + self.roi_height
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return None
        roi = rotated[y1:y2, x1:x2]
        return roi

    def _compute_delta_hu(self, roi: np.ndarray) -> float:
        # 左右 1/3 の平均差でコントラストをざっくり推定
        h, w = roi.shape
        w3 = max(1, w // 3)
        left_mean = float(np.mean(roi[:, :w3]))
        right_mean = float(np.mean(roi[:, -w3:]))
        return abs(right_mean - left_mean)

    def select_rois(self, slices: List, series_name: str, num_rois: int) -> List[ROI]:
        candidates: List[ROI] = []
        if not slices:
            return candidates
        n_slices = len(slices)
        # 層別用（軸方向×水平/垂直様）
        slice_bin_edges = np.linspace(0.0, 1.0, 6)  # 5 bins

        for sl in slices:
            # スライス画像の取得（属性フォールバック）
            src = getattr(sl, "roi_image", None) or getattr(sl, "image", None) or getattr(sl, "array", None)
            if src is None:
                raise AttributeError("DicomSlice does not have roi_image/image/array.")
            img = src.astype(np.float32)

            # 8-bit化
            p2, p98 = np.percentile(img, [2, 98])
            if p98 - p2 < 1e-3:
                img_8 = np.zeros_like(img, dtype=np.uint8)
            else:
                img_clipped = np.clip(img, p2, p98)
                img_8 = (((img_clipped - p2) / (p98 - p2)) * 255.0).astype(np.uint8)

            # Canny & 勾配角
            edges = cv2.Canny(img_8, 50, 150, apertureSize=3)
            sobelx = cv2.Sobel(img_8, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_8, cv2.CV_64F, 0, 1, ksize=3)
            magnitude, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)

            coords = np.column_stack(np.where(edges > 0))
            rng = np.random.default_rng()
            rng.shuffle(coords)

            for (r, c) in coords:
                angle = (float(grad_angle[r, c]) + 90.0) % 180.0  # エッジ角
                diff_h = min(abs(angle - 0.0), abs(angle - 180.0))
                diff_v = abs(angle - 90.0)
                min_diff = min(diff_h, diff_v)
                if min_diff < self.angle_min or min_diff > self.angle_max:
                    continue

                roi_img = self._extract_roi_from_candidate(img, (r, c), angle)
                if roi_img is None:
                    continue

                # --- 一本＆上下貫通 & 中央通過 & ほぼ縦 ---
                ok = True
                if self.require_single_edge:
                    p2_, p98_ = np.percentile(roi_img, [2, 98])
                    if p98_ - p2_ < 1e-3:
                        roi8 = np.zeros_like(roi_img, dtype=np.uint8)
                    else:
                        roi8 = np.clip((roi_img - p2_) / (p98_ - p2_), 0, 1)
                        roi8 = (roi8 * 255).astype(np.uint8)

                    e = cv2.Canny(roi8, 50, 150, apertureSize=3)
                    lines = cv2.HoughLinesP(
                        e, 1, np.pi / 180,
                        threshold=20,
                        minLineLength=int(self.roi_height * self.min_vertical_span_ratio),
                        maxLineGap=5
                    )
                    h_roi, w_roi = roi8.shape
                    vtol = np.deg2rad(self.edge_vertical_tol_deg)
                    valid_lines = []
                    if lines is not None:
                        for x1, y1, x2, y2 in lines.reshape(-1, 4):
                            dx, dy = (x2 - x1), (y2 - y1)
                            ang = abs(np.arctan2(dy, dx))  # 0=横, π/2=縦
                            if abs(ang - np.pi / 2) > vtol:
                                continue
                            ymin, ymax = min(y1, y2), max(y1, y2)
                            span_ok = (ymin <= int(h_roi * 0.05)) and (ymax >= int(h_roi * 0.95))
                            if not span_ok:
                                continue
                            x_mid = 0.5 * (x1 + x2)
                            center_ok = (abs(x_mid - (w_roi - 1) / 2.0) <= w_roi * self.center_tolerance_ratio)
                            if not center_ok:
                                continue
                            valid_lines.append((x1, y1, x2, y2, ang))
                    if len(valid_lines) != 1:
                        ok = False
                    else:
                        _, _, _, _, ang = valid_lines[0]
                        if abs(ang - np.pi / 2) > vtol:
                            ok = False
                if not ok:
                    continue

                delta_hu = self._compute_delta_hu(roi_img)
                if delta_hu < self.delta_hu_threshold:
                    continue

                relative_pos = sl.index / max(1, n_slices - 1)
                axial_bin = 0
                for idx_bin in range(len(slice_bin_edges) - 1):
                    if slice_bin_edges[idx_bin] <= relative_pos < slice_bin_edges[idx_bin + 1]:
                        axial_bin = idx_bin
                        break
                orientation_category = 0 if diff_h <= diff_v else 1
                group = axial_bin * 2 + orientation_category

                candidates.append(
                    ROI(
                        series_name=series_name,
                        slice_index=sl.index,
                        slice_filename=getattr(sl, "filename", ""),
                        roi_image=roi_img.astype(np.float32),
                        orientation_deg=angle,
                        delta_hu=delta_hu,
                        group=group,
                    )
                )
                if len(candidates) >= num_rois * 5:
                    break  # 余裕をもって収集後に層別抽出
            if len(candidates) >= num_rois * 5:
                break

        if not candidates:
            logging.warning(f"No valid ROI candidates found for series {series_name}")
            return []

        # 簡易層別サンプリング
        indices = list(range(len(candidates)))
        groups = [roi.group for roi in candidates]
        # 均等抽出（不足分はランダム補充）
        per_group = max(1, num_rois // max(1, len(set(groups))))
        selected = []
        rng = np.random.default_rng()
        for g in sorted(set(groups)):
            idxs = [i for i in indices if groups[i] == g]
            rng.shuffle(idxs)
            selected.extend(idxs[:per_group])
        if len(selected) < num_rois:
            remain = list(set(indices) - set(selected))
            rng.shuffle(remain)
            selected.extend(remain[: (num_rois - len(selected))])

        selected = selected[:num_rois]
        return [candidates[i] for i in selected]
