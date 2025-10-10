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