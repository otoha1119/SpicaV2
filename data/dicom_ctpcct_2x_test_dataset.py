# data/dicom_ctpcct_2x_test_dataset.py
import os
import math
import torch
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from data.base_dataset import BaseDataset


def _to_hu(ds, arr):
    """DICOMのRescaleSlope/Interceptを使ってHUへ変換（なければそのまま）"""
    try:
        # pydicomのLUTで安全にHU化
        hu = apply_modality_lut(arr, ds).astype(np.float32)
    except Exception:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = arr.astype(np.float32) * slope + inter
    return hu

def _hu_to_unit01(hu):
    """(HU + 1024) / 4095 を [0,1] でクリップ"""
    norm = (hu + 1024.0) / 4095.0
    norm = np.clip(norm, 0.0, 1.0, out=norm)
    return norm.astype(np.float32)

def _pad_to_multiple(img, mod):
    """H,W を mod の倍数へ反射パディング。戻り: padded, (l,r,t,b)"""
    if mod is None or mod <= 1:
        return img, (0, 0, 0, 0)
    h, w = img.shape[-2:]
    pad_h = (mod - (h % mod)) % mod
    pad_w = (mod - (w % mod)) % mod
    pl = pad_w // 2
    pr = pad_w - pl
    pt = pad_h // 2
    pb = pad_h - pt
    if (pl | pr | pt | pb) == 0:
        return img, (0, 0, 0, 0)
    img = np.pad(img, ((0, 0), (pt, pb), (pl, pr)), mode="reflect")
    return img, (pl, pr, pt, pb)

class DicomCtpcct2xTestDataset(BaseDataset):
    """推論専用: 単一DICOMを読み込み、学習時と同じ正規化でテンソル化して返す"""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--input_dicom", type=str, required=True,
                            help="推論入力のDICOMファイルパス")
        parser.add_argument("--output_dicom", type=str, default="results/SR_2x.dcm",
                            help="出力DICOMファイルパス")
        parser.add_argument("--pad_mod", type=int, default=4,
                            help="H,Wをpad_modの倍数へ反射パディング（例:4や8など）。0/1で無効")
        parser.add_argument("--halves_pixel_spacing", action="store_true",
                            help="2×SRに合わせてPixelSpacingを1/2へ更新して保存")
        parser.add_argument("--use_G", type=str, default="A", choices=["A", "B"],
                            help="使用する生成器。A=net_G_A(A→B), B=net_G_B(B→A)")
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.path = opt.input_dicom
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"input_dicom not found: {self.path}")

        ds = pydicom.dcmread(self.path)
        arr = ds.pixel_array  # 元の格納値
        hu = _to_hu(ds, arr)  # HUへ
        norm = _hu_to_unit01(hu)  # [0,1]

        # 形状を (C,H,W) へ（CTは1ch想定）
        if norm.ndim == 2:
            norm = norm[None, :, :]  # (1,H,W)
        elif norm.ndim == 3 and norm.shape[0] != 1:
            # 万一カラーや多chが来たら1chへ平均化
            norm = norm.mean(axis=0, keepdims=True)

        # パディング（Generatorのダウンサンプル段数に応じて4や8を推奨）
        pad_mod = self.opt.pad_mod if getattr(self.opt, "pad_mod", None) else 0
        self.img, (pl, pr, pt, pb) = _pad_to_multiple(norm, pad_mod)
        self.pad = (pl, pr, pt, pb)
        self.orig_hw = tuple(norm.shape[-2:])

        # 保存時に参照したいメタを保持
        self._ref_dicom = ds

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # torch tensorへ
        tensor = torch.from_numpy(self.img.copy())  # (1,H,W), float32 [0,1]
        p = self.path
        paths_list = [p]  # 複数形は必ず list に

        sample = {
            # テンソル（clinical=LR, micro=HR 相当。推論では同一テンソルでOK）
            "A": tensor,
            "clinical": tensor,
            "B": tensor,
            "micro": tensor,

            # パス（単数・複数とも用意。複数形は list 推奨）
            "A_paths": p,
            "B_paths": p,
            "clinical_path": p,
            "micro_path": p,
            "clinical_paths": paths_list,
            "micro_paths": paths_list,

            # 形状・パディング情報（int）
            "pad_l": int(self.pad[0]),
            "pad_r": int(self.pad[1]),
            "pad_t": int(self.pad[2]),
            "pad_b": int(self.pad[3]),
            "orig_h": int(self.orig_hw[0]),
            "orig_w": int(self.orig_hw[1]),
        }

        # DataLoaderのcollateが受け付けないNoneを除去（保険）
        for k in list(sample.keys()):
            if sample[k] is None:
                sample.pop(k)

        return sample
