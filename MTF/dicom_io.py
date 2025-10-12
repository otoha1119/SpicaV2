"""
Utilities for loading and handling DICOM images for MTF analysis.

This module wraps common operations such as reading pixel data,
applying rescale slope and intercept, extracting pixel spacing, and
iterating over series of DICOM files.  It does not attempt to
interpret volumes or reconstruct 3D data; each slice is handled
independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pydicom


@dataclass
class DicomSlice:
    """Simple container for DICOM slice data and metadata."""
    index: int
    filename: Path
    pixel_array: np.ndarray  # 2D array of intensities in Hounsfield units
    pixel_spacing: Tuple[float, float]  # (row_spacing_mm, col_spacing_mm)
    slice_thickness: Optional[float]
    convolution_kernel: Optional[str]


def load_dicom_file(path: Path) -> DicomSlice:
    """
    Load a single DICOM file and return a DicomSlice.

    The pixel data is rescaled to Hounsfield units using RescaleSlope and
    RescaleIntercept when present. Pixel spacing is extracted from the
    PixelSpacing tag. Slice thickness and convolution kernel are also
    recorded when available.
    """
    ds = pydicom.dcmread(path, force=True)
    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception as e:
        logging.error(f"Failed to read pixel data from {path}: {e}")
        raise
    # Apply rescale slope/intercept if present
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    # Pixel spacing
    try:
        spacing = ds.PixelSpacing  # type: ignore
        if isinstance(spacing, (list, tuple)) and len(spacing) == 2:
            row_spacing, col_spacing = float(spacing[0]), float(spacing[1])
        else:
            row_spacing = col_spacing = 1.0
            logging.warning(f"PixelSpacing missing or malformed for {path}")
    except Exception:
        row_spacing = col_spacing = 1.0
        logging.warning(f"PixelSpacing not found for {path}")
    # Slice thickness
    slice_thickness = None
    if hasattr(ds, "SliceThickness"):
        try:
            slice_thickness = float(ds.SliceThickness)
        except Exception:
            slice_thickness = None
    # Convolution kernel
    kernel = None
    if hasattr(ds, "ConvolutionKernel"):
        kernel = str(ds.ConvolutionKernel)

    return DicomSlice(
        index=0,
        filename=path,
        pixel_array=arr,
        pixel_spacing=(row_spacing, col_spacing),
        slice_thickness=slice_thickness,
        convolution_kernel=kernel,
    )


def load_series(directory: Path) -> List[DicomSlice]:
    """
    Load all DICOM files from a directory.

    The files are sorted by InstanceNumber if available; otherwise by filename.
    Each file is converted to Hounsfield units. Pixel spacing and other
    metadata are preserved.
    """
    paths = [p for p in directory.iterdir() if p.is_file() and not p.name.startswith('.')]
    slices = []
    # Read all for sorting
    unsorted: List[Tuple[int, Path, DicomSlice]] = []
    for path in paths:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            instance_num = int(getattr(ds, "InstanceNumber", 0))
        except Exception:
            instance_num = 0
        # read full slice
        try:
            slice_obj = load_dicom_file(path)
        except Exception:
            continue
        unsorted.append((instance_num, path, slice_obj))
    # Sort by instance number then filename
    unsorted.sort(key=lambda x: (x[0], x[1].name))
    for idx, (_, _, slice_obj) in enumerate(unsorted):
        slice_obj.index = idx
        slices.append(slice_obj)
    return slices

def apply_spacing_override(slices, spacing_tuple):
    """
    DICOMスライス配列に対して PixelSpacing を強制上書きする。
    spacing_tuple: (row_mm, col_mm) のタプル（mm単位）
    """
    if not slices:
        return
    try:
        row, col = float(spacing_tuple[0]), float(spacing_tuple[1])
    except Exception:
        import logging
        logging.warning("apply_spacing_override: invalid spacing_tuple; skip.")
        return

    import logging
    n = 0
    for ds in slices:
        try:
            # pydicom Dataset を想定
            ds.PixelSpacing = [row, col]  # DICOMでは [row, col]
            n += 1
        except Exception:
            # 形式が違うなどで失敗したら無視
            pass
    logging.info(f"Applied PixelSpacing override to {n} slices -> [{row}, {col}] mm")


def adjust_pixel_spacing_for_sr(slices, scale: float):
    """
    SRシリーズに対して PixelSpacing を scale 倍『小さく』する（= 解像度 up に合わせる）。
    例: scale=2.0 のとき、0.30mm -> 0.15mm に補正。
    DICOMの PixelSpacing は [row_spacing_mm, col_spacing_mm]。
    """
    import logging

    if not slices:
        logging.warning("adjust_pixel_spacing_for_sr: no SR slices; skip.")
        return
    try:
        s = float(scale)
        if not (s > 0.0):
            logging.warning("adjust_pixel_spacing_for_sr: invalid scale; skip.")
            return
    except Exception:
        logging.warning("adjust_pixel_spacing_for_sr: invalid scale type; skip.")
        return

    n = 0
    for ds in slices:
        try:
            ps = getattr(ds, "PixelSpacing", None)
            if isinstance(ps, (list, tuple)) and len(ps) == 2:
                row = float(ps[0]) / s
                col = float(ps[1]) / s
            else:
                # メタが無い場合は 1.0 mm を仮置きしてから割る（警告）
                row = 1.0 / s
                col = 1.0 / s
                logging.warning(
                    "SR PixelSpacing missing; assumed 1.0 mm before scaling."
                )
            ds.PixelSpacing = [row, col]
            n += 1
        except Exception:
            # 1スライス失敗しても全体は続行
            continue

    logging.info(f"Adjusted SR PixelSpacing by factor {s} for {n} slices.")
