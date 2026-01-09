from __future__ import annotations
import os
import random
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

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
        
        # 類似クロップ設定
        self.use_similar_crop: bool = bool(getattr(opt, 'use_similar_crop', False))
        self.similar_crop_candidates: int = int(getattr(opt, 'similar_crop_candidates', 32))
        self.similar_crop_median_threshold: float = float(getattr(opt, 'similar_crop_median_threshold', 0.15))
        self.similar_crop_max_retries: int = int(getattr(opt, 'similar_crop_max_retries', 3))
        self.similar_crop_max_diff_threshold: float = float(getattr(opt, 'similar_crop_max_diff_threshold', 0.3))
        self.similar_crop_top_pixel_ratio: float = float(getattr(opt, 'similar_crop_top_pixel_ratio', 0.05))
        
        # 画像保存設定（1エポック目1ステップ目のみ）
        self.save_sample_pairs: bool = bool(getattr(opt, 'save_sample_pairs', True))
        self.checkpoints_dir: str = getattr(opt, 'checkpoints_dir', './checkpoints')
        self.experiment_name: str = getattr(opt, 'name', 'SR_CycleGAN')
        self._sample_saved: bool = False  # 1回だけ保存するためのフラグ
        
        # キャッシュ設定
        self.cache_size: int = int(getattr(opt, 'dicom_cache_size', 100))
        self._image_cache: OrderedDict = OrderedDict()

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
    
    def _sample_hr_image(self) -> Tuple[str, np.ndarray]:
        """HR画像をランダムに選択して画像データを返す（類似クロップ用）"""
        hr_path = random.choice(self.hr_slices)[1]
        hr_img = self._get_cached_image(hr_path)
        return hr_path, hr_img
    
    def _save_sample_pair(self, lr_crop: np.ndarray, hr_crop: np.ndarray, lr_path: str, hr_path: str, epoch: int, step: int):
        """1エポック目1ステップ目のペア画像を保存"""
        if not self.save_sample_pairs or self._sample_saved:
            return
        
        if epoch != 1 or step != 1:
            return
        
        # 保存ディレクトリを作成
        save_dir = Path(self.checkpoints_dir) / self.experiment_name / "sample_pairs"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像を[0, 1]から[0, 255]に変換
        lr_img_uint8 = (np.clip(lr_crop, 0, 1) * 255).astype(np.uint8)
        hr_img_uint8 = (np.clip(hr_crop, 0, 1) * 255).astype(np.uint8)
        
        # PIL Imageに変換して保存
        lr_pil = Image.fromarray(lr_img_uint8, mode='L')
        hr_pil = Image.fromarray(hr_img_uint8, mode='L')
        
        # ファイル名からパス情報を取得（簡易版）
        lr_name = Path(lr_path).stem
        hr_name = Path(hr_path).stem
        
        lr_pil.save(save_dir / f"LR_epoch{epoch}_step{step}_{lr_name}.png")
        hr_pil.save(save_dir / f"HR_epoch{epoch}_step{step}_{hr_name}.png")
        
        # サイドバイサイドで保存
        combined = Image.new('L', (lr_crop.shape[1] + hr_crop.shape[1], max(lr_crop.shape[0], hr_crop.shape[0])))
        combined.paste(lr_pil, (0, 0))
        combined.paste(hr_pil, (lr_crop.shape[1], 0))
        combined.save(save_dir / f"pair_epoch{epoch}_step{step}_LR-{lr_name}_HR-{hr_name}.png")
        
        self._sample_saved = True
        print(f"[Dataset] Sample pair saved to {save_dir}")

    def _get_cached_image(self, path: str) -> np.ndarray:
        """LRUキャッシュを使用して画像を取得"""
        if self.cache_size <= 0:
            # キャッシュ無効の場合
            return dio.read_normalized_pixels(path)
        
        # キャッシュに存在する場合
        if path in self._image_cache:
            # 最新アクセスとして先頭に移動
            self._image_cache.move_to_end(path)
            return self._image_cache[path].copy()
        
        # キャッシュに存在しない場合、読み込んでキャッシュに追加
        img = dio.read_normalized_pixels(path)
        
        # キャッシュサイズ制限チェック
        if len(self._image_cache) >= self.cache_size:
            # 最も古いエントリを削除（LRU）
            self._image_cache.popitem(last=False)
        
        # キャッシュに追加（コピーを保存）
        self._image_cache[path] = img.copy()
        
        return img

    # ---------- PyTorch Dataset API ----------
    def __len__(self) -> int:
        # Always return INT
        #print("[DEBUG] __len__ called in DicomCtpcct2xDataset")
        return int(self._len)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # 1) Load LR slice (normalized [0,1]) - キャッシュ使用
        lr_path = self._sample_lr_path(index)
        lr_img = self._get_cached_image(lr_path)  # HxW float32 in [0,1]
        lr_mask = dio.compute_body_mask(lr_img, self.body_thresh_norm) if self.use_body_mask else None
        lr_crop = dio.crop_random(
            lr_img, self.lr_patch,
            require_mask=lr_mask,
            min_coverage=self.min_body_coverage if self.use_body_mask else 0.0
        )

        # 2) Load HR slice (normalized [0,1]) - キャッシュ使用
        if self.use_similar_crop:
            # 類似クロップを使用
            hr_path, hr_img = self._sample_hr_image()
            hr_mask = dio.compute_body_mask(hr_img, self.body_thresh_norm) if self.use_body_mask else None
            hr_crop, retry_count = dio.crop_similar_with_retry(
                hr_img,
                lr_crop,
                self.hr_patch,
                num_candidates=self.similar_crop_candidates,
                require_mask=hr_mask,
                min_coverage=self.min_body_coverage if self.use_body_mask else 0.0,
                max_image_retries=self.similar_crop_max_retries,
                median_diff_threshold=self.similar_crop_median_threshold,
                max_diff_threshold=self.similar_crop_max_diff_threshold,
                top_pixel_ratio=self.similar_crop_top_pixel_ratio
            )
        else:
            # 従来のランダムクロップ
            hr_path = self._sample_hr_path(index)
            hr_img = self._get_cached_image(hr_path)
            hr_mask = dio.compute_body_mask(hr_img, self.body_thresh_norm) if self.use_body_mask else None
            hr_crop = dio.crop_random(
                hr_img, self.hr_patch,
                require_mask=hr_mask,
                min_coverage=self.min_body_coverage if self.use_body_mask else 0.0
            )

        # 3) To torch (C=1)
        lr_t = torch.from_numpy(lr_crop).unsqueeze(0).float()
        hr_t = torch.from_numpy(hr_crop).unsqueeze(0).float()

        # 4) 1エポック目1ステップ目の画像保存（epochとstepは外部から取得する必要があるため、ここではindex=0の場合のみ）
        # 実際のepoch/stepはtrainerから渡す必要があるが、簡易的にindex=0で保存
        if index == 0 and not self._sample_saved:
            # epochとstepは実際にはtrainerから取得する必要があるが、ここでは仮の値を使用
            # 実際の実装では、trainerからコールバックで呼び出すか、別の方法を検討
            pass  # 後でtrainer側で実装

        # 5) Return with multiple key aliases for compatibility
        return {
            # CycleGAN default keys
            'A': lr_t, 'B': hr_t,
            'A_paths': lr_path, 'B_paths': hr_path,

            # SR-CycleGAN medical aliases
            'clinical': lr_t, 'micro': hr_t,
            'clinical_paths': lr_path, 'micro_paths': hr_path,
            
            # 画像保存用（後でtrainerから使用）
            'lr_crop_np': lr_crop,
            'hr_crop_np': hr_crop,
        }
