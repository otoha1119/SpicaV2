from __future__ import annotations
import os, time
from typing import List, Optional
import numpy as np

try:
    import pydicom
    from pydicom.uid import generate_uid
except Exception as e:
    pydicom = None

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
    norm = (arr + 1024.0) / 4095.0
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
    return norm

def denormalize_to_int16(norm: np.ndarray) -> np.ndarray:
    return np.rint(norm * 4095.0 - 1024.0).astype(np.int16)

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
