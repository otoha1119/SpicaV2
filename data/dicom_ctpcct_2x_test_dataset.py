# data/dicom_ctpcct_2x_test_dataset.py
# -------------------------------
# DICOM オブジェクトを返さず、完全に安全な Dataset に修正

import os
import torch
import numpy as np
import pydicom
from data.base_dataset import BaseDataset


class DicomCtpcct2xTestDataset(BaseDataset):
    """推論専用：学習と同じ正規化で 1 枚 DICOM を読み込む"""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--input_dicom", type=str, required=True)
        parser.add_argument("--output_dicom", type=str, default="results/SR_2x.dcm")
        parser.add_argument("--use_G", type=str, default="A")
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.path = opt.input_dicom

        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"input_dicom not found: {self.path}")

        # ---- DICOM 読み込み ----
        ds = pydicom.dcmread(self.path)
        arr = ds.pixel_array.astype(np.int32)

        # ---- 学習と同じ正規化 ----
        norm = (arr + 2048.0) / 6143.0
        norm = np.clip(norm, 0.0, 1.0).astype(np.float32)

        # ---- (1,H,W) ----
        if norm.ndim == 2:
            norm = norm[None, :, :]
        else:
            norm = norm.mean(axis=0, keepdims=True)

        self.norm = norm

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.norm.copy())  # (1,H,W)
        p = self.path
        paths_list = [p]

        return {
            # テンソル（学習側のインターフェースに合わせる）
            "A": tensor,
            "B": tensor,
            "clinical": tensor,   # ← MedicalCycleGANModel が参照
            "micro": tensor,      # ← MedicalCycleGANModel が参照

            # パス類（互換性のために一通り持たせておく）
            "A_paths": p,
            "B_paths": p,
            "clinical_path": p,
            "micro_path": p,
            "clinical_paths": paths_list,
            "micro_paths": paths_list,
        }
