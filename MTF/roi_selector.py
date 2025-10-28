"""
ROI selection for natural high‑contrast edges in CT slices.

This module implements an automated approach to extract a set of
rectangular regions of interest (ROIs) around slanted high‑contrast edges
from CT slices.  The goal is to identify candidate edges in each
slice that meet certain geometric and contrast criteria, then sample
ROIs oriented along the detected edges.  The design follows the
recommendations outlined in ISO 12233 for slanted‑edge MTF estimation
and the literature on CT resolution assessment.

The primary function exposed by this module is ``select_rois``.  It
takes a list of DicomSlice objects and returns a list of ROIs along
with metadata describing their location and orientation.

Edge detection uses Canny followed by computation of gradient
orientation from Sobel filters.  Candidate edge pixels whose
orientation is within a specified angular range (e.g. 5°–15° relative
to horizontal or vertical) are considered.  For each candidate, a
small patch is rotated such that the edge is vertical, and a
rectangular ROI is extracted.  The mean intensities on either side of
the edge are compared to ensure sufficient contrast (ΔHU threshold).
ROIs are stratified by axial location and orientation when sampling to
provide a balanced distribution across the dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2  # type: ignore

from .dicom_io import DicomSlice
from .utils import bin_edges_to_groups, stratified_sample


@dataclass
class ROI:
    """A data container holding an ROI and associated metadata."""
    series_name: str  # 'LR', 'SR', or 'HR'
    slice_index: int
    slice_filename: Path
    roi_image: np.ndarray  # shape (height, width), orientation‑aligned
    orientation_deg: float  # original edge orientation in degrees (0=horiz)
    delta_hu: float  # contrast difference across edge (absolute)
    group: int  # stratification group index


class ROISelector:
    """Select ROIs from a list of DicomSlice objects."""

    def __init__(
        self,
        roi_width: int = 30,
        roi_height: int = 100,
        angle_min: float = 5.0,
        angle_max: float = 15.0,
        delta_hu_threshold: float = 200.0,
    ):
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.delta_hu_threshold = delta_hu_threshold

    def _extract_roi_from_candidate(
        self,
        img: np.ndarray,
        center: Tuple[int, int],
        orientation: float,
    ) -> np.ndarray:
        """
        Extract a rotated ROI from the image centred at `center` and oriented
        such that the detected edge becomes vertical.  The ROI is returned
        axis‑aligned (height x width) with dimensions (roi_height, roi_width).

        Parameters
        ----------
        img : np.ndarray
            The input image (2D array).
        center : Tuple[int, int]
            The row and column index of the candidate edge pixel in the
            original image.
        orientation : float
            The edge orientation in degrees relative to the x‑axis (0° means
            horizontal edge).  The patch is rotated by (90° − orientation)
            so that the edge becomes vertical.

        Returns
        -------
        np.ndarray
            A rotated ROI of shape (roi_height, roi_width) oriented such
            that the edge is vertical.
        """
        row, col = center
        # Determine patch size: ensure rotated ROI fits inside patch.
        patch_side = int(np.ceil(np.hypot(self.roi_width, self.roi_height))) + 4
        # Clip patch centre to avoid going out of bounds
        half = patch_side // 2
        # Get patch from original image
        h, w = img.shape
        # Coordinates for subpixel extraction are (x,y) where x is col, y is row
        if not (half <= col < w - half and half <= row < h - half):
            return None  # type: ignore
        patch = cv2.getRectSubPix(img, (patch_side, patch_side), (float(col), float(row)))
        # Rotate patch: rotation angle = 90 − orientation
        rot_angle = 90.0 - orientation
        M = cv2.getRotationMatrix2D((patch_side / 2.0, patch_side / 2.0), rot_angle, 1.0)
        rotated = cv2.warpAffine(
            patch,
            M,
            (patch_side, patch_side),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # Extract aligned ROI from rotated patch (width x height)
        roi = cv2.getRectSubPix(
            rotated,
            (self.roi_width, self.roi_height),
            (patch_side / 2.0, patch_side / 2.0),
        )
        return roi

    def _compute_delta_hu(self, roi: np.ndarray) -> float:
        """Compute the absolute HU difference between left and right sides of the ROI.

        The ROI is assumed to be oriented such that the edge is vertical
        and runs down the middle of the ROI.  We compute the mean
        intensity in the leftmost 1/3 and rightmost 1/3 of the ROI and
        return the absolute difference.
        """
        if roi is None:
            return 0.0
        h, w = roi.shape
        if w < 3:
            return 0.0
        third = w // 3
        left = roi[:, :third]
        right = roi[:, -third:]
        mean_left = float(np.mean(left))
        mean_right = float(np.mean(right))
        return abs(mean_right - mean_left)

    def select_rois(
        self,
        slices: List[DicomSlice],
        series_name: str,
        num_rois: int,
    ) -> List[ROI]:
        """
        Select a set of ROIs from a list of DICOM slices.

        Parameters
        ----------
        slices : list of DicomSlice
            The input slices for a single series (LR, SR or HR).
        series_name : str
            Name of the series (used in ROI metadata).
        num_rois : int
            Desired number of ROIs to extract.

        Returns
        -------
        list of ROI
            The selected ROIs.

        Notes
        -----
        - Candidates are generated from edges detected via Canny and
          filtered by gradient orientation.  The process stops once a
          multiple of `num_rois` candidates have been considered to
          prevent unbounded processing time.
        - The final selection uses stratified sampling across three
          axial regions (bottom/middle/top) and two orientation
          categories (horizontal‑like or vertical‑like) to balance the
          distribution of ROIs.  If stratification fails to fill all
          requested slots, a random subset of the remaining candidates
          is used.
        """
        if num_rois <= 0:
            return []
        candidates: List[ROI] = []
        # Determine slice bins for stratification
        n_slices = len(slices)
        if n_slices == 0:
            return []
        # We map slice index to one of three axial bins: [0,1/3), [1/3,2/3), [2/3,1]
        slice_bin_edges = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 + 1e-6]
        # Process slices sequentially and gather candidates until we have enough
        max_candidates = num_rois * 10  # gather at most 10× more candidates than needed
        for sl in slices:
            if len(candidates) >= max_candidates:
                break
            img = sl.pixel_array
            # Convert to 8‑bit for edge detection; scale and clamp values
            # We'll scale intensities to [0,255] using percentiles to be robust to outliers
            p2, p98 = np.percentile(img, [2, 98])
            if p98 - p2 < 1e-3:
                img_8 = np.zeros_like(img, dtype=np.uint8)
            else:
                img_clipped = np.clip(img, p2, p98)
                img_8 = (((img_clipped - p2) / (p98 - p2)) * 255.0).astype(np.uint8)
            # Edge detection
            edges = cv2.Canny(img_8, 50, 150, apertureSize=3)
            # Gradient orientation (edge orientation is gradient+90)
            sobelx = cv2.Sobel(img_8, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_8, cv2.CV_64F, 0, 1, ksize=3)
            # gradient orientation in degrees (0<=angle<360)
            magnitude, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
            # Iterate through edge pixels randomly to avoid cluster bias
            coords = np.column_stack(np.where(edges > 0))
            rng = np.random.default_rng()
            rng.shuffle(coords)
            for (r, c) in coords:
                # Determine edge orientation relative to horizontal
                # The orientation of the edge is perpendicular to gradient direction
                angle = (float(grad_angle[r, c]) + 90.0) % 180.0  # wrap to [0,180)
                # Compute deviation from nearest horizontal (0 deg) and vertical (90 deg)
                diff_h = min(abs(angle - 0.0), abs(angle - 180.0))
                diff_v = abs(angle - 90.0)
                # Minimum deviation from either axis
                min_diff = min(diff_h, diff_v)
                if min_diff < self.angle_min or min_diff > self.angle_max:
                    continue
                # Determine orientation category: 0 for horizontal‑like, 1 for vertical‑like
                orientation_category = 0 if diff_h <= diff_v else 1
                # Extract ROI oriented such that edge becomes vertical
                roi_img = self._extract_roi_from_candidate(img, (r, c), angle)
                if roi_img is None:
                    continue
                delta_hu = self._compute_delta_hu(roi_img)
                if delta_hu < self.delta_hu_threshold:
                    continue
                # Determine axial bin
                relative_pos = sl.index / max(1, n_slices - 1)
                axial_bin = None
                for idx, (low, high) in enumerate(zip(slice_bin_edges[:-1], slice_bin_edges[1:])):
                    if low <= relative_pos < high:
                        axial_bin = idx
                        break
                if axial_bin is None:
                    axial_bin = len(slice_bin_edges) - 2
                # Combined group index: axial_bin * 2 + orientation_category
                group = axial_bin * 2 + orientation_category
                candidates.append(
                    ROI(
                        series_name=series_name,
                        slice_index=sl.index,
                        slice_filename=sl.filename,
                        roi_image=roi_img.astype(np.float32),
                        orientation_deg=angle,
                        delta_hu=delta_hu,
                        group=group,
                    )
                )
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        # If no candidates found, return empty
        if not candidates:
            logging.warning(f"No valid ROI candidates found for series {series_name}")
            return []
        # Stratified sampling
        indices = list(range(len(candidates)))
        groups = [roi.group for roi in candidates]
        selected_indices = stratified_sample(indices, groups, num_rois)
        # If not enough selected, fill with random
        if len(selected_indices) < num_rois:
            remaining = list(set(indices) - set(selected_indices))
            rng = np.random.default_rng()
            extra = rng.choice(remaining, size=min(num_rois - len(selected_indices), len(remaining)), replace=False)
            selected_indices.extend(extra.tolist())
        # Clip to requested number
        selected_indices = selected_indices[:num_rois]
        selected_rois = [candidates[i] for i in selected_indices]
        return selected_rois