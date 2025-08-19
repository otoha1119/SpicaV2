from __future__ import annotations
import os, numpy as np, torch
from typing import Dict, Any
from data.base_dataset import BaseDataset
from util import dicom_io as dio

class DicomCtpcct2xTestDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--input_dicom', type=str, required=True)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.input_path = opt.input_dicom
        if not os.path.isfile(self.input_path):
            raise RuntimeError(f"input_dicom not found: {self.input_path}")
        self.norm_img = dio.read_normalized_pixels(self.input_path)

    def __len__(self): return 1

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img = self.norm_img.astype(np.float32)
        t = torch.from_numpy(img).unsqueeze(0).float()
        return {'A': t, 'A_paths': self.input_path, 'clinical': t, 'clinical_path': self.input_path}
