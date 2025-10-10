# -*- coding: utf-8 -*-
"""
Batch DICOM -> SR(2x) inference.

- Reuses the same model for all files.
- For each input DICOM, builds a tiny one-item dataset and runs test(), then saves a DICOM
  that preserves the reference metadata and (optionally) halves PixelSpacing.
"""
import os
import os.path as osp
from glob import glob
import argparse
import torch
import numpy as np
import pydicom
from pydicom.uid import generate_uid

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torch.nn as nn

# ---------- utils copied from single-inference ----------
def unit01_to_hu(unit: np.ndarray) -> np.ndarray:
    """[0,1] -> HU。学習時の規約：HU = unit*4095 - 1024"""
    hu = unit * 4095.0 - 1024.0
    return hu.astype(np.float32)

def hu_to_stored_like_ref(hu_img: np.ndarray, ref_ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """参照DICOMの RescaleSlope/Intercept に合わせて格納値へ戻す。"""
    slope = float(getattr(ref_ds, "RescaleSlope", 1.0))
    inter = float(getattr(ref_ds, "RescaleIntercept", 0.0))
    stored = (hu_img - inter) / (slope if slope != 0 else 1.0)

    bits_stored = int(getattr(ref_ds, "BitsStored", 16))
    signed = int(getattr(ref_ds, "PixelRepresentation", 1))  # 1: signed, 0: unsigned
    if signed:
        minv = -(1 << (bits_stored - 1))
        maxv = (1 << (bits_stored - 1)) - 1
        dtype = np.int16 if bits_stored <= 16 else np.int32
    else:
        minv = 0
        maxv = (1 << bits_stored) - 1
        dtype = np.uint16 if bits_stored <= 16 else np.uint32

    stored = np.clip(np.rint(stored), minv, maxv).astype(dtype)
    return stored

def update_uids(ds: pydicom.dataset.FileDataset) -> None:
    """新規SOP/Series UIDを発行"""
    new_sop = generate_uid()
    new_series = generate_uid()
    ds.SOPInstanceUID = new_sop
    ds.SeriesInstanceUID = new_series
    if hasattr(ds, "file_meta"):
        ds.file_meta.MediaStorageSOPInstanceUID = new_sop

def save_like_reference(unit_img_2d: np.ndarray,
                        ref_ds: pydicom.dataset.FileDataset,
                        out_path: str,
                        halves_pixel_spacing: bool = False) -> None:
    """2×SR([0,1]) を参照DICOMのメタを継承して保存。"""
    hu = unit01_to_hu(unit_img_2d)
    stored = hu_to_stored_like_ref(hu, ref_ds)

    ds = ref_ds.copy()

    H, W = stored.shape
    ds.Rows = int(H)
    ds.Columns = int(W)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    if not hasattr(ds, "BitsAllocated"):
        ds.BitsAllocated = 16
    if not hasattr(ds, "BitsStored"):
        ds.BitsStored = 16
    if not hasattr(ds, "HighBit"):
        ds.HighBit = ds.BitsStored - 1
    if not hasattr(ds, "PixelRepresentation"):
        ds.PixelRepresentation = 1  # signed

    if halves_pixel_spacing and hasattr(ds, "PixelSpacing"):
        try:
            ps = [float(x) for x in ds.PixelSpacing]
            ds.PixelSpacing = [ps[0] * 0.5, ps[1] * 0.5]
        except Exception:
            pass

    update_uids(ds)

    ds.PixelData = stored.tobytes()
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    ds.save_as(out_path)

def pick_sr_visual(visuals: dict) -> torch.Tensor:
    """生成結果のキー揺れに備えて安全に選ぶ。"""
    candidates = ["fake_B", "fake_micro", "fake_B_full", "fake_B_sr", "SR"]
    for k in candidates:
        if k in visuals:
            return visuals[k]
    # fallback: 最大テンソル
    best = None
    best_hw = -1
    for v in visuals.values():
        if torch.is_tensor(v):
            t = v
        elif isinstance(v, (list, tuple)) and len(v) > 0 and torch.is_tensor(v[0]):
            t = v[0]
        else:
            continue
        if t.ndim >= 2:
            h = t.shape[-2]; w = t.shape[-1]
            if h * w > best_hw:
                best_hw = h * w
                best = t
    if best is None:
        raise RuntimeError("No valid visuals found for SR output.")
    return best

def crop_unpad_after_2x(out_tensor: torch.Tensor,
                        pad_l: int, pad_r: int, pad_t: int, pad_b: int) -> torch.Tensor:
    """2×SR後の出力から、入力前に施した反射パディングを2倍相当にして除去。"""
    if out_tensor.ndim == 4:
        _, _, H, W = out_tensor.shape
        t = out_tensor
    elif out_tensor.ndim == 3:
        _, H, W = out_tensor.shape
        t = out_tensor.unsqueeze(0)
    else:
        raise ValueError("Output tensor must be 3D or 4D.")
    l = pad_l * 2; r = pad_r * 2
    tt = pad_t * 2; b = pad_b * 2
    t = t[:, :, tt:H - b if b > 0 else H, l:W - r if r > 0 else W]
    return t.squeeze(0) if out_tensor.ndim == 3 else t

def _pick_ref_path(data: dict) -> str:
    """data(dict)から参照DICOMのパスを安全に取り出す（複数形は先頭）。"""
    for k in ("A_paths", "clinical_paths", "A_path", "clinical_path",
              "micro_paths", "micro_path", "B_paths", "B_path"):
        if k in data and data[k] is not None:
            p = data[k]
            if isinstance(p, (list, tuple)) and len(p) > 0:
                return p[0]
            return p
    raise RuntimeError("Reference DICOM path not found in data dict.")

def _load_specific_generator(model, opt) -> None:
    """/{which_epoch}_net_G_{A|B}.pth を実体(nn.Module)にロード"""
    use = getattr(opt, "use_G", "A")
    ckpt = osp.join(opt.checkpoints_dir, opt.name, f"{opt.which_epoch}_net_G_{use}.pth")
    if not osp.isfile(ckpt):
        raise FileNotFoundError(f"Generator checkpoint not found: {ckpt}")

    target = getattr(model, f"netG_{use}", None) or getattr(model, "netG", None)
    if target is None:
        raise AttributeError(f"model has no netG_{use} nor netG")

    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    real_module = target.module if isinstance(target, nn.DataParallel) else target

    # 常に module. を剥がす（実体は素の nn.Module）
    sd = { (k.replace("module.", "", 1) if k.startswith("module.") else k): v
           for k, v in sd.items() }

    real_module.load_state_dict(sd, strict=False)

# ---------- core ----------
def build_model(opt):
    """モデルを構築して重みをロードして eval にする"""
    # dataset_mode 未指定ならテスト用既定
    if not getattr(opt, "dataset_mode", None) or opt.dataset_mode == "single":
        opt.dataset_mode = "dicom_ctpcct_2x_test"

    # setup前に自動ロードを止めて初期化のみ
    model = create_model(opt)
    orig = getattr(model, "model_names", [])
    model.model_names = []
    model.setup(opt)
    model.model_names = orig

    # 学習済みGを手動ロード
    _load_specific_generator(model, opt)

    model.eval()
    return model

def infer_one(model, opt, input_path: str, output_path: str):
    """1ファイル推論（モデルは再利用）"""
    # 入出力を一時的に差し替え
    setattr(opt, "input_dicom", input_path)
    setattr(opt, "output_dicom", output_path)

    dataset = create_dataset(opt)
    data = next(iter(dataset))

    with torch.no_grad():
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        out = pick_sr_visual(visuals)

        if out.ndim == 4:
            out = out[0]
        if out.ndim == 2:
            out = out.unsqueeze(0)
        out = torch.clamp(out, 0.0, 1.0).cpu()

        out = crop_unpad_after_2x(
            out,
            int(data.get("pad_l", 0)),
            int(data.get("pad_r", 0)),
            int(data.get("pad_t", 0)),
            int(data.get("pad_b", 0)),
        )

        out2d = out[0].numpy() if out.shape[0] == 1 else out.mean(dim=0).numpy()

    # サイズ検証（2x 固定）
    orig_h, orig_w = int(data["orig_h"]), int(data["orig_w"])
    exp_h, exp_w = orig_h * 2, orig_w * 2
    if out2d.shape != (exp_h, exp_w):
        raise RuntimeError(
            f"Expected 2x output {(exp_h, exp_w)} but got {out2d.shape} @ {input_path}"
        )

    # 保存
    ref_path = _pick_ref_path(data)
    ref_ds = pydicom.dcmread(ref_path)
    save_like_reference(out2d, ref_ds, output_path,
                        halves_pixel_spacing=bool(getattr(opt, "halves_pixel_spacing", False)))

def list_dicoms(root: str):
    """root 以下の DICOM を列挙（*.dcm, *.dicom, 大文字小文字）"""
    pats = ["*.dcm", "*.DCM", "*.dicom", "*.DICOM"]
    files = []
    for dirpath, _, _ in os.walk(root):
        for pat in pats:
            files.extend(glob(osp.join(dirpath, pat)))
    files = sorted(set(files))
    return files

def main():
    # ===== 1) 入出力ディレクトリ取得（まず環境変数を優先） =====
    in_dir = os.environ.get("INPUT_DIR", None)
    out_dir = os.environ.get("OUTPUT_DIR", None)

    # ===== 2) argparse の必須(--input_dicom)回避のためのダミー注入 =====
    import sys
    argv_backup = list(sys.argv)
    if "--input_dicom" not in sys.argv:
        sys.argv += ["--input_dicom", "/dev/null"]
    if "--output_dicom" not in sys.argv:
        sys.argv += ["--output_dicom", "/dev/null"]

    # CUDA が見えない/使えない場合は CPU にフォールバック
    try:
        has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        has_cuda = False
    if not has_cuda:
        if "--gpu_ids" in sys.argv:
            gi = sys.argv.index("--gpu_ids")
            if gi + 1 < len(sys.argv):
                sys.argv[gi + 1] = "-1"
        else:
            sys.argv += ["--gpu_ids", "-1"]

    # ===== 3) TestOptions をパース（ダミーのおかげで落ちない） =====
    opt = TestOptions().parse()

    # パース後は argv を元に戻しておく（副作用防止）
    sys.argv = argv_backup

    # ===== 4) 実行時の最小上書き =====
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    if not getattr(opt, "which_epoch", None):
        setattr(opt, "which_epoch", str(getattr(opt, "epoch", "latest")))

    # TestOptions に --input_dir/--output_dir がある場合の保険（なければ環境変数を使う）
    if not in_dir:
        in_dir = getattr(opt, "input_dir", None)
    if not out_dir:
        out_dir = getattr(opt, "output_dir", None)

    # 必須チェック
    if not in_dir or not out_dir:
        raise SystemExit("ERROR: 入出力ディレクトリが指定されていません。（INPUT_DIR / OUTPUT_DIR を設定してください）")

    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Input dir: {in_dir}")
    print(f"[INFO] Output dir: {out_dir}")

    # ===== 5) モデルを1回だけ構築・重みロード =====
    model = build_model(opt)

    # ===== 6) 入力ディレクトリ配下の DICOM を列挙 =====
    paths = list_dicoms(in_dir)
    if not paths:
        raise SystemExit(f"No DICOM files found under: {in_dir}")

    # ===== 7) 推論ループ =====
    for ipath in paths:
        base = osp.splitext(osp.basename(ipath))[0]
        opath = osp.join(out_dir, f"{base}_SR2x.dcm")
        infer_one(model, opt, ipath, opath)
        print(f"[OK] {ipath} -> {opath}")

    print(f"[DONE] All results saved under: {out_dir}")




if __name__ == "__main__":
    main()
