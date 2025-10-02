#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_dicom_ctpcct_dataset.py
"""
import argparse
import os
import sys
import json
import random
from typing import List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pydicom
    from pydicom.uid import generate_uid
except Exception:
    pydicom = None

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
DATA_DIR = os.path.join(THIS_DIR, "data")
if os.path.isdir(DATA_DIR) and DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

try:
    from data.dicom_ctpcct_2x_dataset import DicomCtpcct2xDataset
except Exception:
    from dicom_ctpcct_2x_dataset import DicomCtpcct2xDataset  # type: ignore

dio = None
try:
    from util import dicom_io as dio  # type: ignore
except Exception:
    dio = None


def set_seed(seed: int):
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_histogram(values: np.ndarray, out_png: str, title: str = "", bins: int = 256):
    fig = plt.figure()
    plt.hist(values, bins=bins, range=(0.0, 1.0))
    plt.title(title)
    plt.xlabel("Normalized intensity (0–1)")
    plt.ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_stats_csv(stats: List[dict], out_csv: str):
    import csv
    if not stats:
        return
    keys = sorted(stats[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in stats:
            w.writerow(r)


def try_read_normalized_pixels(dcm_path: str) -> np.ndarray:
    if dio is not None and hasattr(dio, "read_normalized_pixels"):
        return dio.read_normalized_pixels(dcm_path)
    if pydicom is None:
        raise RuntimeError("pydicom が見つかりません。pip install pydicom を実行してください。")
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.int32)
    norm = (arr + 1024.0) / 4095.0
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
    return norm


def try_denormalize_to_int16(img01: np.ndarray) -> np.ndarray:
    if dio is not None and hasattr(dio, "denormalize_to_int16"):
        return dio.denormalize_to_int16(img01)
    arr = np.rint(np.clip(img01, 0.0, 1.0) * 4095.0 - 1024.0).astype(np.int16)
    return arr

def try_save_dicom_like(ref_path: str, img_any: np.ndarray, out_path: str):
    """
    DICOM保存ユーティリティ。
    - util.dicom_io.save_dicom_like があれば (norm_img, ref_path, out_path) の順で呼ぶ。
      norm_img は [0,1] の float32 を想定。失敗したら pydicom フォールバック。
    - フォールバックは int16 をそのまま PixelData に書く（位置タグの厳密復元は行わない）。
    """
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # まずは util 側を試す（正しい順のみ）
    if dio is not None and hasattr(dio, "save_dicom_like"):
        # img_any -> [0,1] float32 に揃える
        if np.issubdtype(img_any.dtype, np.floating):
            norm = np.clip(img_any, 0.0, 1.0).astype(np.float32)
        else:
            norm = ((img_any.astype(np.float32) + 1024.0) / 4095.0)
            norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
        try:
            # ★ 引数順を固定：(norm_img, ref_path, out_path)
            return dio.save_dicom_like(norm, ref_path, out_path)
        except Exception as e:
            # util 側が使えなければフォールバックへ
            pass

    # ---- フォールバック: pydicom で int16 保存 ----
    if pydicom is None:
        raise RuntimeError("pydicom が見つかりません。pip install pydicom を実行してください。")

    # 0-1 float の場合は逆正規化して int16 へ。int系ならそのまま int16 化。
    if np.issubdtype(img_any.dtype, np.floating):
        int16_img = np.rint(np.clip(img_any, 0.0, 1.0) * 4095.0 - 1024.0).astype(np.int16)
    else:
        int16_img = img_any.astype(np.int16)

    ds = pydicom.dcmread(ref_path)
    try:
        from pydicom.uid import generate_uid
        ds.SOPInstanceUID = generate_uid(prefix=None)
    except Exception:
        pass

    h, w = int16_img.shape
    ds.Rows = h
    ds.Columns = w
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    ds.PixelData = int16_img.tobytes()
    ds.save_as(out_path)


def _build_dataset_opts_from_dataset(is_train: bool):
    """Dataset側のデフォルト引数を取得"""
    parser = argparse.ArgumentParser(add_help=False)
    parser = DicomCtpcct2xDataset.modify_commandline_options(parser, is_train=is_train)
    ds_opt = parser.parse_args([])
    return ds_opt


def _inject_required_base_opts(ds_opt, lr_root: str, hr_root: str, is_train: bool):
    """BaseDataset が期待する最低限の属性を補う"""
    # dataroot は必須（BaseDatasetが参照）
    try:
        common = os.path.commonpath([lr_root, hr_root])
    except Exception:
        common = lr_root
    setattr(ds_opt, "dataroot", getattr(ds_opt, "dataroot", common))
    # isTrain / phase も補う（無ければ）
    setattr(ds_opt, "isTrain", getattr(ds_opt, "isTrain", is_train))
    setattr(ds_opt, "phase", getattr(ds_opt, "phase", "train" if is_train else "test"))
    # serial_batches は False 既定で良い
    setattr(ds_opt, "serial_batches", getattr(ds_opt, "serial_batches", False))
    # max_dataset_size 既定
    setattr(ds_opt, "max_dataset_size", getattr(ds_opt, "max_dataset_size", 1e9))


def export_cropped_patches_as_dicom(opt, out_dir: str):
    set_seed(opt.seed)
    ensure_dir(out_dir)
    out_ct = os.path.join(out_dir, "clinical")
    out_pcct = os.path.join(out_dir, "pcct")
    ensure_dir(out_ct)
    ensure_dir(out_pcct)

    ds_opt = _build_dataset_opts_from_dataset(is_train=True)
    # 代入で上書き
    setattr(ds_opt, "lr_root", opt.lr_root)
    setattr(ds_opt, "hr_root", opt.hr_root)
    setattr(ds_opt, "lr_patch", opt.lr_patch)
    setattr(ds_opt, "hr_patch", opt.hr_patch)
    setattr(ds_opt, "hr_oversample_ratio", opt.hr_oversample_ratio)
    setattr(ds_opt, "use_body_mask", bool(opt.use_body_mask) or getattr(ds_opt, "use_body_mask", False))
    setattr(ds_opt, "body_thresh_norm", opt.body_thresh_norm)
    setattr(ds_opt, "min_body_coverage", opt.min_body_coverage)
    setattr(ds_opt, "epoch_size", max(opt.n_patches, 1))
    setattr(ds_opt, "fast_scan", bool(opt.fast_scan) or getattr(ds_opt, "fast_scan", False))
    _inject_required_base_opts(ds_opt, opt.lr_root, opt.hr_root, is_train=True)

    dataset = DicomCtpcct2xDataset(ds_opt)

    n = min(opt.n_patches, len(dataset))
    print(f"[CROP] exporting {n} LR/HR patches as DICOM ...")

    stats = []
    for i in range(n):
        item = dataset[i]
        lr = item["A"].numpy().squeeze()
        hr = item["B"].numpy().squeeze()
        lr_path = item["A_paths"]
        hr_path = item["B_paths"]

        lr_int16 = try_denormalize_to_int16(lr)
        hr_int16 = try_denormalize_to_int16(hr)

        out_lr = os.path.join(out_ct, f"patch_{i:04d}.dcm")
        out_hr = os.path.join(out_pcct, f"patch_{i:04d}.dcm")
        try_save_dicom_like(lr_path, lr_int16, out_lr)
        try_save_dicom_like(hr_path, hr_int16, out_hr)

        stats.append({
            "idx": i,
            "lr_min": float(lr.min()), "lr_max": float(lr.max()), "lr_mean": float(lr.mean()), "lr_std": float(lr.std()),
            "hr_min": float(hr.min()), "hr_max": float(hr.max()), "hr_mean": float(hr.mean()), "hr_std": float(hr.std()),
            "lr_path": lr_path, "hr_path": hr_path
        })

    save_stats_csv(stats, os.path.join(out_dir, "patch_stats.csv"))
    print(f"[CROP] done. -> {out_dir}")


def export_full_histograms(opt, out_dir: str):
    ensure_dir(out_dir)

    ds_opt = _build_dataset_opts_from_dataset(is_train=False)
    setattr(ds_opt, "lr_root", opt.lr_root)
    setattr(ds_opt, "hr_root", opt.hr_root)
    setattr(ds_opt, "fast_scan", bool(opt.fast_scan) or getattr(ds_opt, "fast_scan", False))
    _inject_required_base_opts(ds_opt, opt.lr_root, opt.hr_root, is_train=False)

    dummy = DicomCtpcct2xDataset(ds_opt)

    lr_list = getattr(dummy, "lr_slices", [])
    hr_list = getattr(dummy, "hr_slices", [])

    n_lr = min(opt.n_full, len(lr_list))
    n_hr = min(opt.n_full, len(hr_list))

    print(f"[HIST] using {n_lr} LR full-slices, {n_hr} HR full-slices")

    all_lr_vals = []
    all_hr_vals = []
    per_image_stats = []

    for i in range(n_lr):
        _, p = lr_list[i]
        img01 = try_read_normalized_pixels(p).astype(np.float32)
        all_lr_vals.append(img01.ravel())
        per_image_stats.append({
            "domain": "LR",
            "idx": i, "path": p,
            "min": float(img01.min()), "max": float(img01.max()),
            "mean": float(img01.mean()), "std": float(img01.std())
        })

    for i in range(n_hr):
        _, p = hr_list[i]
        img01 = try_read_normalized_pixels(p).astype(np.float32)
        all_hr_vals.append(img01.ravel())
        per_image_stats.append({
            "domain": "HR",
            "idx": i, "path": p,
            "min": float(img01.min()), "max": float(img01.max()),
            "mean": float(img01.mean()), "std": float(img01.std())
        })

    if all_lr_vals:
        lr_vals = np.concatenate(all_lr_vals, axis=0)
        save_histogram(lr_vals, os.path.join(out_dir, "hist_LR.png"), title="LR normalized histogram", bins=256)

    if all_hr_vals:
        hr_vals = np.concatenate(all_hr_vals, axis=0)
        save_histogram(hr_vals, os.path.join(out_dir, "hist_HR.png"), title="HR normalized histogram", bins=256)

    save_stats_csv(per_image_stats, os.path.join(out_dir, "full_stats.csv"))
    print(f"[HIST] done. -> {out_dir}")


def export_full_roundtrip(opt, out_dir: str):
    ensure_dir(out_dir)
    out_ct = os.path.join(out_dir, "clinical")
    out_pcct = os.path.join(out_dir, "pcct")
    ensure_dir(out_ct)
    ensure_dir(out_pcct)

    ds_opt = _build_dataset_opts_from_dataset(is_train=False)
    setattr(ds_opt, "lr_root", opt.lr_root)
    setattr(ds_opt, "hr_root", opt.hr_root)
    setattr(ds_opt, "fast_scan", bool(opt.fast_scan) or getattr(ds_opt, "fast_scan", False))
    _inject_required_base_opts(ds_opt, opt.lr_root, opt.hr_root, is_train=False)

    dummy = DicomCtpcct2xDataset(ds_opt)
    lr_list = getattr(dummy, "lr_slices", [])
    hr_list = getattr(dummy, "hr_slices", [])

    n_lr = min(opt.n_full, len(lr_list))
    n_hr = min(opt.n_full, len(hr_list))

    diff_stats = []

    def _roundtrip_one(path: str, out_dir_domain: str, idx: int, domain: str):
        img01 = try_read_normalized_pixels(path).astype(np.float32)
        rec = try_denormalize_to_int16(img01)
        out_path = os.path.join(out_dir_domain, f"recon_{idx:04d}.dcm")
        try_save_dicom_like(path, rec, out_path)

        if pydicom is not None:
            ds = pydicom.dcmread(path)
            orig = ds.pixel_array.astype(np.int32)
            diff = (rec.astype(np.int32) - orig).astype(np.int32)
            mae = float(np.mean(np.abs(diff)))
            mx = int(np.max(np.abs(diff)))
            diff_stats.append({
                "domain": domain,
                "idx": idx,
                "path": path,
                "mae_abs": mae,
                "max_abs": mx,
                "rec_min": int(rec.min()), "rec_max": int(rec.max()),
                "orig_min": int(orig.min()), "orig_max": int(orig.max())
            })

    print(f"[ROUNDTRIP] exporting {n_lr} LR + {n_hr} HR full-size recon DICOM ...")

    for i in range(n_lr):
        _, p = lr_list[i]
        _roundtrip_one(p, out_ct, i, "LR")

    for i in range(n_hr):
        _, p = hr_list[i]
        _roundtrip_one(p, out_pcct, i, "HR")

    save_stats_csv(diff_stats, os.path.join(out_dir, "roundtrip_diff_stats.csv"))
    print(f"[ROUNDTRIP] done. -> {out_dir}")


def build_argparser():
    ap = argparse.ArgumentParser(description="Verify dicom_ctpcct_2x_dataset.py normalization/cropping pipeline")
    ap.add_argument("--lr_root", type=str, required=True, help="LR(CT) DICOM root directory")
    ap.add_argument("--hr_root", type=str, required=True, help="HR(PCCT) DICOM root directory")
    ap.add_argument("--out_dir", type=str, default="./out_verify_dataset", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n_patches", type=int, default=40, help="Num cropped patches to export for each domain")
    ap.add_argument("--n_full", type=int, default=20, help="Num full-size slices to process for each domain")

    ap.add_argument("--lr_patch", type=int, default=98, help="LR patch size")
    ap.add_argument("--hr_patch", type=int, default=196, help="HR patch size")
    ap.add_argument("--hr_oversample_ratio", type=float, default=1.0)
    ap.add_argument("--use_body_mask", action="store_true", help="Enable body mask")
    ap.add_argument("--body_thresh_norm", type=float, default=0.1)
    ap.add_argument("--min_body_coverage", type=float, default=0.3)
    ap.add_argument("--fast_scan", action="store_true")

    return ap


def main():
    ap = build_argparser()
    opt = ap.parse_args()
    set_seed(opt.seed)

    out_root = os.path.abspath(opt.out_dir)
    ensure_dir(out_root)

    export_cropped_patches_as_dicom(opt, os.path.join(out_root, "1_cropped_patches_dicom"))
    export_full_histograms(opt, os.path.join(out_root, "2_full_hist"))
    export_full_roundtrip(opt, os.path.join(out_root, "3_full_roundtrip_dicom"))

    with open(os.path.join(out_root, "config.json"), "w") as f:
        json.dump(vars(opt), f, indent=2)

    print("[DONE] All exports completed.")


if __name__ == "__main__":
    main()
