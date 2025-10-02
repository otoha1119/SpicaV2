# -*- coding: utf-8 -*-
"""
単一 DICOM を SR-CycleGAN の G_A(Clinical->Micro) で超解像推論するスクリプト
 - 入出力: DICOM → PNG/DICOM
 - 学習時の正規化に合わせて [0,1] スケールを想定
 - 生成器出力が Tanh の場合は --assume_tanh で [-1,1]→[0,1] に戻す
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image

# 学習プロジェクト配下のモジュール（models/networks.py など）にパスを通す
import sys
sys.path.append('/workspace')
sys.path.append('/workspace/models')

# 学習時のネットワーク実装をそのまま使う
from models.networks import get_norm_layer
from models.networks import Clinical2MicroGenerator

# DICOM I/O はプロジェクトの util を優先。なければ pydicom でフォールバック
try:
    from util import dicom_io as dio
    _USE_PROJECT_DICOM = True
except Exception:
    _USE_PROJECT_DICOM = False
    import pydicom
    from pydicom.uid import generate_uid


def _read_dicom_as_norm01(path: str) -> (np.ndarray, dict):
    """DICOM を読み、[0,1] 画像と必要メタを返す"""
    if _USE_PROJECT_DICOM:
        img = dio.read_normalized_pixels(path)  # HxW float32 in [0,1]
        meta = {"raw_path": path}
        return img, meta
    else:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        # 最低限の正規化（学習側の実装に合わせるなら dio を使うのが確実）
        # 既知: 論文実装では (val + 1024)/4095 想定。ここでは min-max 代替。
        arr = (arr - arr.min()) / max(1e-6, (arr.max() - arr.min()))
        return arr, {"ds": ds}


def _save_png_01(img01: np.ndarray, out_path: str):
    """[0,1] の ndarray を 8bit グレースケール PNG に保存"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    img8 = np.clip(img01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(img8).save(out_path)


def _save_dicom_01(img01: np.ndarray, meta: dict, out_path: str):
    """[0,1] 画像を DICOM で保存（簡易版）"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    if _USE_PROJECT_DICOM:
        # util.dicom_io に保存関数があればそれを使う（なければ下の簡易保存）
        try:
            dio.save_normalized_to_dicom(img01, meta.get("raw_path", None), out_path)
            return
        except Exception:
            pass  # なければ簡易保存にフォールバック

    # 簡易保存（元 DICOM メタがあれば継承）
    ds = meta.get("ds", None)
    if ds is None:
        ds = pydicom.Dataset()
        ds.file_meta = pydicom.Dataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.SOPClassUID = generate_uid()
        ds.SOPInstanceUID = generate_uid()
        ds.is_little_endian = True
        ds.is_implicit_VR = False

    img16 = np.clip(img01 * 4095.0 + 0.5, 0, 4095).astype(np.uint16)  # 12bit 相当
    ds.Rows, ds.Columns = int(img16.shape[0]), int(img16.shape[1])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsStored = 12
    ds.BitsAllocated = 16
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelData = img16.tobytes()
    pydicom.dcmwrite(out_path, ds)


def _to_tensor_01(img01: np.ndarray) -> torch.Tensor:
    """[0,1] HxW → torch.FloatTensor [1,1,H,W]"""
    t = torch.from_numpy(img01).float().unsqueeze(0).unsqueeze(0)
    return t


def _to_numpy_01(t: torch.Tensor) -> np.ndarray:
    """torch.Tensor [1,1,H,W] → [0,1] HxW"""
    t = t.detach().cpu().float().squeeze(0).squeeze(0)
    return np.clip(t.numpy(), 0.0, 1.0)


def _maybe_denorm_from_tanh(out: torch.Tensor) -> torch.Tensor:
    """Tanh出力([-1,1])を[0,1]へ戻す"""
    return (out + 1.0) * 0.5


def load_generator(ckpt_path: str, device: torch.device):
    """学習済み G_A (Clinical2Micro) をロード（strict=Falseで安全読み込み）"""
    norm_layer = get_norm_layer('instance')

    # 学習側の定義に合わせたコンストラクタ（入力/出力1ch、ngf=64がデフォルト想定）
    try:
        netG = Clinical2MicroGenerator(input_nc=1, output_nc=1, ngf=64,
                                       norm_layer=norm_layer, use_dropout=False)
    except TypeError:
        # 引数位置指定のみ受ける古い定義へのフォールバック
        netG = Clinical2MicroGenerator(1, 1, 64, norm_layer=norm_layer, use_dropout=False)

    netG.to(device)
    netG.eval()

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    # DataParallel で保存された 'module.' 接頭辞を除去
    cleaned = {}
    for k, v in state.items():
        if k.startswith('module.'):
            cleaned[k[len('module.'):]] = v
        else:
            cleaned[k] = v

    # 多少の不一致は許容
    missing, unexpected = netG.load_state_dict(cleaned, strict=False)
    if missing:
        print("[WARN] missing keys:", missing)
    if unexpected:
        print("[WARN] unexpected keys:", unexpected)

    return netG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dicom', type=str, required=True)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--name', type=str, default='SR_CycleGAN')
    parser.add_argument('--epoch', type=str, default='latest')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--out_png', type=str, default='./results/sr.png')
    parser.add_argument('--out_dicom', type=str, default='./results/sr.dcm')
    parser.add_argument('--assume_tanh', action='store_true', help='出力がTanhの場合に[0,1]へ戻す')
    args = parser.parse_args()

    # GPU 選択
    if args.gpu_ids == '-1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 入力読込（[0,1]）
    img01, meta = _read_dicom_as_norm01(args.input_dicom)
    print(f"[INFO] input shape: {img01.shape[1]}x{img01.shape[0]}, min={img01.min():.3f}, max={img01.max():.3f}")

    # チェックポイント
    ckpt_path = os.path.join(args.checkpoints_dir, args.name, f'{args.epoch}_net_G_A.pth')
    if args.epoch == 'latest':
        ckpt_path = os.path.join(args.checkpoints_dir, args.name, 'latest_net_G_A.pth')
    print(f"[INFO] ckpt: {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    # 生成器ロード
    netG = load_generator(ckpt_path, device)

    # 推論
    with torch.no_grad():
        tin = _to_tensor_01(img01).to(device)
        tout = netG(tin)
        if args.assume_tanh:
            tout = _maybe_denorm_from_tanh(tout)
        sr01 = _to_numpy_01(tout)

    # 保存
    _save_png_01(sr01, args.out_png)
    print(f"[INFO] saved PNG -> {args.out_png}")
    _save_dicom_01(sr01, meta, args.out_dicom)
    print(f"[INFO] saved DICOM -> {args.out_dicom}")


if __name__ == '__main__':
    main()
