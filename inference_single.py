# medicaltest_dicom.py
import os
import torch
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from collections import OrderedDict
import torch.nn as nn

# ---------- 正規化/逆正規化（学習規約と一致） ----------
def unit01_to_hu(unit: np.ndarray) -> np.ndarray:
    """[0,1] -> HU。学習時の規約：HU = unit*4095 - 1024"""
    hu = unit * 4095.0 - 1024.0
    return hu.astype(np.float32)

def hu_to_stored_like_ref(hu_img: np.ndarray, ref_ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    参照DICOM(ref_ds)の RescaleSlope/Intercept に合わせて格納値へ戻す。
    PixelRepresentation, BitsStored に合わせてクリップ。
    """
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
    """
    2×SRの出力（[0,1]）を、参照DICOMのメタを継承して保存。
    halves_pixel_spacing=True のとき PixelSpacing を 1/2 に更新。
    """
    # 逆正規化 unit->[HU] -> 参照のSlope/Interceptで格納値へ
    hu = unit01_to_hu(unit_img_2d)
    stored = hu_to_stored_like_ref(hu, ref_ds)

    ds = ref_ds.copy()

    # 画素配列とサイズ
    H, W = stored.shape
    ds.Rows = int(H)
    ds.Columns = int(W)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # ビット関連（不足時は安全側）
    if not hasattr(ds, "BitsAllocated"):
        ds.BitsAllocated = 16
    if not hasattr(ds, "BitsStored"):
        ds.BitsStored = 16
    if not hasattr(ds, "HighBit"):
        ds.HighBit = ds.BitsStored - 1
    if not hasattr(ds, "PixelRepresentation"):
        ds.PixelRepresentation = 1  # signed

    # PixelSpacing を 1/2（2×SR）に更新（指定時）
    if halves_pixel_spacing and hasattr(ds, "PixelSpacing"):
        try:
            ps = [float(x) for x in ds.PixelSpacing]
            ds.PixelSpacing = [ps[0] * 0.5, ps[1] * 0.5]
        except Exception:
            pass

    # UIDs更新（衝突回避）
    update_uids(ds)

    # ピクセル書き込み & 保存
    ds.PixelData = stored.tobytes()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ds.save_as(out_path)
    print(f"[OK] Saved: {out_path}")

def pick_sr_visual(visuals: dict) -> torch.Tensor:
    """
    生成結果のキー揺れに備えて安全に選ぶ。
    優先: fake_B -> fake_micro -> fake_B_full -> fake_B_sr -> SR
    フォールバック: 最も大きい2D/3Dテンソル
    """
    candidates = ["fake_B", "fake_micro", "fake_B_full", "fake_B_sr", "SR"]
    for k in candidates:
        if k in visuals:
            return visuals[k]
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
    """
    2×SR後の出力から、入力前に施した反射パディングを2倍相当にして除去。
    入力: (N,C,H,W) または (C,H,W)
    """
    if out_tensor.ndim == 4:
        _, _, H, W = out_tensor.shape
        t = out_tensor
    elif out_tensor.ndim == 3:
        _, H, W = out_tensor.shape
        t = out_tensor.unsqueeze(0)
    else:
        raise ValueError("Output tensor must be 3D or 4D.")

    # 2×なのでパディングも2倍して除去
    l = pad_l * 2; r = pad_r * 2
    tt = pad_t * 2; b = pad_b * 2
    t = t[:, :, tt:H - b if b > 0 else H, l:W - r if r > 0 else W]
    return t.squeeze(0) if out_tensor.ndim == 3 else t

def _pick_ref_path(data: dict) -> str:
    """data(dict)から参照DICOMのパスを安全に取り出す（複数形は先頭を使う）。"""
    for k in ("A_paths", "clinical_paths", "A_path", "clinical_path",
              "micro_paths", "micro_path", "B_paths", "B_path"):
        if k in data and data[k] is not None:
            p = data[k]
            if isinstance(p, (list, tuple)) and len(p) > 0:
                return p[0]
            return p
    raise RuntimeError("Reference DICOM path not found in data dict.")

def _load_specific_generator(model, opt) -> None:
    """
    --use_G A|B に応じて /{which_epoch}_net_G_A/B.pth を手動ロードする。
    （setup中の自動ロードは抑止してある）
    """
    use = getattr(opt, "use_G", "A")   # "A" or "B"
    ckpt = os.path.join(opt.checkpoints_dir, opt.name,
                        f"{opt.which_epoch}_net_G_{use}.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Generator checkpoint not found: {ckpt}")

    # medical_cycle_gan は netG_A / netG_B を持つ想定
    target = getattr(model, f"netG_{use}", None)
    if target is None:
        target = getattr(model, "netG", None)
    if target is None:
        raise AttributeError(f"model has no netG_{use} nor netG")

    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    

    # 読み込み先の“実体”を決定（DataParallelなら .module）
    real_module = target.module if isinstance(target, nn.DataParallel) else target

    sd = {
        (k.replace("module.", "", 1) if k.startswith("module.") else k): v
        for k, v in sd.items()
    }

    # パラメータ総ノルムで“重みが入ったか”も可視化
    def _pnorm(m):
        try:
            return sum(p.detach().float().abs().sum().item() for p in m.parameters())
        except Exception:
            return -1


    ret = real_module.load_state_dict(sd, strict=False)

    miss = getattr(ret, "missing_keys", [])
    unexp = getattr(ret, "unexpected_keys", [])
    


def main():
    # 1) オプションをパース（CLIの --model/--dataset_mode などを尊重）
    opt = TestOptions().parse()

    # 2) 既定の上書き（最小限。model は上書きしない）
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # dataset_mode 未指定のときだけ既定化
    if not getattr(opt, "dataset_mode", None) or opt.dataset_mode == "single":
        opt.dataset_mode = "dicom_ctpcct_2x_test"

    # ★CLIを尊重★ which_epoch が無ければ epoch から補完（上書きしない）
    if not getattr(opt, "which_epoch", None):
        setattr(opt, "which_epoch", str(getattr(opt, "epoch", "latest")))

    

    # # チェックポイントとエポック（手動ロードで使用）
    # opt.checkpoints_dir = "/workspace/checkpoints"
    # opt.name = "SR_CycleGAN"
    # opt.epoch = "167"
    # setattr(opt, "which_epoch", "167")

    # 3) データセット＆モデルを作成
    dataset = create_dataset(opt)
    model = create_model(opt)
    
    with torch.no_grad():
        net = getattr(model, "netG_A")  # A→B をテスト（B→Aなら netG_B）
        dev = next(net.parameters()).device
        dummy = torch.randn(1, 1, 16, 16, device=dev)
        out_d = net(dummy)

    # 4) 自動ロード抑止 → setup（重みはまだ読まない）
    orig_names = getattr(model, "model_names", [])
    model.model_names = []   # setup 内の load_networks をスキップ
    model.setup(opt)         # ネット初期化のみ
    model.model_names = orig_names

    # 5) 手動で 167 の G_A/B を読み込み（--use_G A|B）
    _load_specific_generator(model, opt)

    model.eval()

    # 6) 推論（1サンプル）
    data = next(iter(dataset))
    with torch.no_grad():
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        out = pick_sr_visual(visuals)  # Tensor

        # 形状正規化 (C,H,W)
        if out.ndim == 4:
            out = out[0]
        if out.ndim == 2:
            out = out.unsqueeze(0)
        out = torch.clamp(out, 0.0, 1.0).cpu()

        # 入力前に施した反射パディングを 2× で除去（2倍固定前提）
        out = crop_unpad_after_2x(
            out,
            int(data.get("pad_l", 0)),
            int(data.get("pad_r", 0)),
            int(data.get("pad_t", 0)),
            int(data.get("pad_b", 0)),
        )

        # (H,W) へ
        if out.shape[0] == 1:
            out2d = out[0].numpy()
        else:
            out2d = out.mean(dim=0).numpy()  # 念のため多chは平均

    # 7) 期待サイズチェック（2×固定）
    orig_h, orig_w = int(data["orig_h"]), int(data["orig_w"])
    exp_h, exp_w = orig_h * 2, orig_w * 2
    if out2d.shape != (exp_h, exp_w):
        # 2×固定前提なので、ズレたら明示エラー（ネット構成のズレを早期検知）
        raise RuntimeError(
            f"Expected 2x output {(exp_h, exp_w)} but got {out2d.shape}. "
            f"Check netG/upscale settings or your checkpoint direction."
        )

    # 8) 保存（2×固定なので halves_pixel_spacing=True でPixelSpacingを1/2に）
    out_path = getattr(opt, "output_dicom", "results/SR_2x.dcm")
    halves_ps = bool(getattr(opt, "halves_pixel_spacing", False))

    ref_path = _pick_ref_path(data)
    ref_ds = pydicom.dcmread(ref_path)
    save_like_reference(out2d, ref_ds, out_path, halves_pixel_spacing=halves_ps)

if __name__ == "__main__":
    main()
