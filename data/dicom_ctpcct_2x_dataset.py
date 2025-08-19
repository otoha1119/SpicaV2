from __future__ import annotations
import os, random
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset

from data.base_dataset import BaseDataset
from util import dicom_io as dio

class DicomCtpcct2xDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--lr_root', type=str, default='/workspace/DataSet/ImageCAS')
        parser.add_argument('--hr_root', type=str, default='/workspace/DataSet/photonCT/PhotonCT1024v2')
        parser.add_argument('--lr_patch', type=int, default=98)
        parser.add_argument('--hr_patch', type=int, default=196)
        parser.add_argument('--hr_oversample_ratio', type=float, default=1.0)
        parser.add_argument('--use_body_mask', action='store_true')
        parser.add_argument('--body_thresh_norm', type=float, default=0.1)
        parser.add_argument('--min_body_coverage', type=float, default=0.3)
        parser.add_argument('--epoch_size', type=int, default=0)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.lr_root = opt.lr_root
        self.hr_root = opt.hr_root
        self.lr_patch = int(opt.lr_patch)
        self.hr_patch = int(opt.hr_patch)
        self.hr_oversample_ratio = float(opt.hr_oversample_ratio)
        self.use_body_mask = bool(opt.use_body_mask)
        self.body_thresh_norm = float(opt.body_thresh_norm)
        self.min_body_coverage = float(opt.min_body_coverage)
        self.epoch_size = int(opt.epoch_size)

        self.lr_series = self._scan_series(self.lr_root)
        self.hr_series = self._scan_series(self.hr_root)
        self.lr_slices = [(pid,p) for pid,ps in self.lr_series.items() for p in ps]
        self.hr_slices = [(pid,p) for pid,ps in self.hr_series.items() for p in ps]
        if len(self.lr_slices)==0: raise RuntimeError(f"No LR DICOM under {self.lr_root}")
        if len(self.hr_slices)==0: raise RuntimeError(f"No HR DICOM under {self.hr_root}")
        self._len = self.epoch_size if self.epoch_size>0 else max(len(self.lr_slices), len(self.hr_slices))

    def _scan_series(self, root_dir: str):
        series = {}
        if not os.path.isdir(root_dir): return series
        for p in sorted(os.listdir(root_dir)):
            pdir = os.path.join(root_dir, p)
            if not os.path.isdir(pdir): continue
            paths = dio.list_dicom_files_recursive(pdir)
            if len(paths)==0: continue
            series[p] = dio.sort_series_paths(paths)
        return series

    def __len__(self): return self._len

    def _sample_lr_path(self, idx: int) -> str:
        return self.lr_slices[idx % len(self.lr_slices)][1]

    def _sample_hr_path(self, idx: int) -> str:
        if self.hr_oversample_ratio <= 1.0:
            return random.choice(self.hr_slices)[1]
        if random.random() < min(1.0, self.hr_oversample_ratio - 1.0) / max(1.0, self.hr_oversample_ratio):
            return random.choice(self.hr_slices)[1]
        return self.hr_slices[idx % len(self.hr_slices)][1]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        lr_path = self._sample_lr_path(index)
        hr_path = self._sample_hr_path(index)
        lr_img = dio.read_normalized_pixels(lr_path)
        hr_img = dio.read_normalized_pixels(hr_path)
        lr_mask = dio.compute_body_mask(lr_img, self.body_thresh_norm) if self.use_body_mask else None
        hr_mask = dio.compute_body_mask(hr_img, self.body_thresh_norm) if self.use_body_mask else None
        lr_crop = dio.crop_random(lr_img, self.lr_patch, lr_mask, self.min_body_coverage if self.use_body_mask else 0.0)
        hr_crop = dio.crop_random(hr_img, self.hr_patch, hr_mask, self.min_body_coverage if self.use_body_mask else 0.0)
        lr_t = torch.from_numpy(lr_crop).unsqueeze(0).float()
        hr_t = torch.from_numpy(hr_crop).unsqueeze(0).float()
        return {
            'A': lr_t, 'B': hr_t,
            'A_paths': lr_path, 'B_paths': hr_path,
            'clinical': lr_t, 'micro': hr_t,
            'clinical_path': lr_path, 'micro_path': hr_path,
        }
