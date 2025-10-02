from __future__ import annotations
import os
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.base_dataset import BaseDataset
from util import dicom_io as dio


class DicomCtpcct2xDataset(BaseDataset):
    """
    Unpaired LR/HR DICOM dataset for 2x super-resolution (medical CT -> PCCT).
    - Directory structure (recursive scan under each patient folder):
        LR_ROOT/<patient>/**/*.dcm
        HR_ROOT/<patient>/**/*.dcm
    - Normalization: (val + 1024) / 4095 -> [0,1]  (no HU conversion)
    - Independent random crops for LR/HR (unpaired learning)
    - Optional lightweight body mask constraint to avoid background-only crops
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # Roots
        parser.add_argument('--lr_root', type=str, default='/workspace/DataSet/ImageCAS',
                            help='Root directory for LR DICOM tree (patients as subdirs).')
        parser.add_argument('--hr_root', type=str, default='/workspace/DataSet/photonCT/PhotonCT1024v2',
                            help='Root directory for HR DICOM tree (patients as subdirs).')

        # Patch sizes (2x SR: e.g., LR=98, HR=196)
        parser.add_argument('--lr_patch', type=int, default=98, help='LR patch size (pixels).')
        parser.add_argument('--hr_patch', type=int, default=196, help='HR patch size (pixels).')

        # Sampling balance
        parser.add_argument('--hr_oversample_ratio', type=float, default=1.0,
                            help='>1.0 increases probability of sampling HR slices (unpaired).')

        # Body mask options (OFF by default)
        parser.add_argument('--use_body_mask', action='store_true',
                            help='Enable body mask and require minimum body coverage in crops.')
        parser.add_argument('--body_thresh_norm', type=float, default=0.1,
                            help='Threshold in normalized scale for body mask.')
        parser.add_argument('--min_body_coverage', type=float, default=0.3,
                            help='Required fraction of body pixels inside a crop when mask enabled.')

        # Epoch size override (optional)
        parser.add_argument('--epoch_size', type=int, default=0,
                            help='If >0, overrides dataset length per epoch (int).')
        parser.add_argument('--fast_scan', action='store_true',
                            help='Speed up scanning: skip per-file DICOM header reads; sort by filename only.')


        return parser

    def __init__(self, opt):
        """
        Build slice lists for LR/HR by scanning patient subdirectories.
        """
        super().__init__(opt)

        # Options
        self.fast_scan: bool = bool(getattr(opt, 'fast_scan', False))
        self.lr_root: str = opt.lr_root
        self.hr_root: str = opt.hr_root
        self.lr_patch: int = int(opt.lr_patch)
        self.hr_patch: int = int(opt.hr_patch)
        self.hr_oversample_ratio: float = float(opt.hr_oversample_ratio)

        self.use_body_mask: bool = bool(opt.use_body_mask)
        self.body_thresh_norm: float = float(opt.body_thresh_norm)
        self.min_body_coverage: float = float(opt.min_body_coverage)
        self.epoch_size: int = int(getattr(opt, 'epoch_size', 0))

        # Scan LR/HR series
        self.lr_series: Dict[str, List[str]] = self._scan_series(self.lr_root)
        self.hr_series: Dict[str, List[str]] = self._scan_series(self.hr_root)

        # Flatten slice lists ([(patient_id, path), ...])
        self.lr_slices: List[Tuple[str, str]] = [
            (pid, p) for pid, paths in self.lr_series.items() for p in paths
        ]
        self.hr_slices: List[Tuple[str, str]] = [
            (pid, p) for pid, paths in self.hr_series.items() for p in paths
        ]

        if len(self.lr_slices) == 0:
            raise RuntimeError(f"No LR DICOM slices found under {self.lr_root}")
        if len(self.hr_slices) == 0:
            raise RuntimeError(f"No HR DICOM slices found under {self.hr_root}")

        # Dataset length (must be INT)
        self._len: int = int(self.epoch_size) if self.epoch_size > 0 else max(len(self.lr_slices), len(self.hr_slices))
        
        #print(f"[DEBUG] dataset length = {len(self)} (type: {type(self._len)})")
        


    # ---------- helpers ----------
    def _scan_series(self, root_dir: str) -> Dict[str, List[str]]:
        out = {}
        if not os.path.isdir(root_dir):
            return out
        for patient in sorted(os.listdir(root_dir)):
            pdir = os.path.join(root_dir, patient)
            if not os.path.isdir(pdir):
                continue
            paths = dio.list_dicom_files_recursive(pdir)
            if len(paths) == 0:
                continue
            # ★ ここを切り替え
            if self.fast_scan:
                paths = sorted(paths)  # ヘッダは読まず、名前順
            else:
                paths = dio.sort_series_paths(paths)  # 従来: ヘッダを読んでZ方向でソート
            out[patient] = paths
        return out

    def _sample_lr_path(self, idx: int) -> str:
        # Deterministic round-robin for LR
        return self.lr_slices[idx % len(self.lr_slices)][1]

    def _sample_hr_path(self, idx: int) -> str:
        # Oversampling: mix deterministic index with random pick depending on ratio
        if self.hr_oversample_ratio <= 1.0:
            # simple random HR slice
            return random.choice(self.hr_slices)[1]
        # Probability mass for extra random samples
        prob = min(1.0, self.hr_oversample_ratio - 1.0) / max(1.0, self.hr_oversample_ratio)
        if random.random() < prob:
            return random.choice(self.hr_slices)[1]
        return self.hr_slices[idx % len(self.hr_slices)][1]

    # ---------- PyTorch Dataset API ----------
    def __len__(self) -> int:
        # Always return INT
        #print("[DEBUG] __len__ called in DicomCtpcct2xDataset")
        return int(self._len)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # 1) Load LR slice (normalized [0,1])
        lr_path = self._sample_lr_path(index)
        lr_img = dio.read_normalized_pixels(lr_path)  # HxW float32 in [0,1]
        lr_mask = dio.compute_body_mask(lr_img, self.body_thresh_norm) if self.use_body_mask else None
        lr_crop = dio.crop_random(
            lr_img, self.lr_patch,
            require_mask=lr_mask,
            min_coverage=self.min_body_coverage if self.use_body_mask else 0.0
        )

        # 2) Load HR slice (normalized [0,1])
        hr_path = self._sample_hr_path(index)
        hr_img = dio.read_normalized_pixels(hr_path)
        hr_mask = dio.compute_body_mask(hr_img, self.body_thresh_norm) if self.use_body_mask else None
        hr_crop = dio.crop_random(
            hr_img, self.hr_patch,
            require_mask=hr_mask,
            min_coverage=self.min_body_coverage if self.use_body_mask else 0.0
        )

        # 3) To torch (C=1)
        lr_t = torch.from_numpy(lr_crop).unsqueeze(0).float()
        hr_t = torch.from_numpy(hr_crop).unsqueeze(0).float()

        # 4) Return with multiple key aliases for compatibility
        return {
            # CycleGAN default keys
            'A': lr_t, 'B': hr_t,
            'A_paths': lr_path, 'B_paths': hr_path,

            # SR-CycleGAN medical aliases
            'clinical': lr_t, 'micro': hr_t,
            'clinical_paths': lr_path, 'micro_paths': hr_path,
        }
