# -*- coding: utf8 -*-
"""TensorBoard visualizer (Visualizer2)

- 1段目: real_A, fake_B, real_B, fake_A を横一列に並べた画像を出力（表示用に自動リサイズ）
- 2段目: 損失のスカラー（log_losses から記録）
- 3段目: それ以外の各画像名（real_A, fake_B, rec_A, ...）を個別グリッドで出力
"""

import os
from collections import OrderedDict
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


class Visualizer2:
    """TensorBoard Visualizer"""

    def __init__(self, opt):
        """
        Args:
            opt: TrainOptions で parse されたオプション
        """
        self.opt = opt
        # TensorBoard の出力先（必要なら "runs" -> "logs_tb" に変更可）
        self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, "runs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # デバイス
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and (opt.gpu_ids is None or opt.gpu_ids != "-1") else "cpu")

        # 1段目に並べるキーの順序
        self.top_row_names = ["real_A", "fake_B", "real_B", "fake_A"]

    # --------- 画像ユーティリティ ----------
    @staticmethod
    def _to_bchw(t: torch.Tensor) -> torch.Tensor:
        """任意の Tensor を [B, C, H, W] に揃える（Cは1chを想定・多chは先頭1chを利用）"""
        if t is None:
            return None
        if t.dim() == 2:
            # [H, W] -> [1,1,H,W]
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:
            # [C,H,W] -> [1,C,H,W]
            t = t.unsqueeze(0)
        elif t.dim() == 4:
            # そのまま
            pass
        else:
            raise ValueError(f"Unsupported tensor shape for image: {t.shape}")

        if t.shape[1] > 1:
            t = t[:, :1, :, :]
        return t

    def _resize_to(self, t: torch.Tensor, size_hw: tuple) -> torch.Tensor:
        """表示用にバイリニアでサイズを合わせる（[B,1,H,W]想定）"""
        th, tw = size_hw
        if (t.shape[-2], t.shape[-1]) == (th, tw):
            return t
        return F.interpolate(t, size=(th, tw), mode="bilinear", align_corners=False)

    # --------- 公開API ----------
    def display_current_results(self, visuals: Dict[str, Any], epoch: int, step: int):
        """
        visuals: model.get_current_visuals() が返す dict（Tensor）
        epoch: 現在のエポック
        step: グローバルステップ（イテレーション）
        """
        if visuals is None or not isinstance(visuals, (dict, OrderedDict)) or len(visuals) == 0:
            return

        # ----- 1) トップ行（real_A, fake_B, real_B, fake_A を横一列）-----
        row_imgs: List[torch.Tensor] = []
        target_h, target_w = None, None

        # まずは real_B のサイズがあればそれに揃える
        rb = visuals.get("real_B", None)
        if isinstance(rb, torch.Tensor):
            rb = self._to_bchw(rb)
            target_h, target_w = int(rb.shape[-2]), int(rb.shape[-1])

        # 無ければ、候補の中で最大サイズに揃える
        if target_h is None:
            sizes = []
            for name in self.top_row_names:
                t = visuals.get(name, None)
                if isinstance(t, torch.Tensor):
                    t = self._to_bchw(t)
                    sizes.append((int(t.shape[-2]), int(t.shape[-1])))
            if sizes:
                target_h, target_w = max(sizes, key=lambda x: x[0] * x[1])
            else:
                target_h, target_w = 128, 128  # フォールバック

        for name in self.top_row_names:
            t = visuals.get(name, None)
            if isinstance(t, torch.Tensor):
                t = self._to_bchw(t)
                t = self._resize_to(t, (target_h, target_w))
            else:
                t = torch.zeros(1, 1, target_h, target_w, device=self.device)
            row_imgs.append(t)

        row_batch = torch.cat(row_imgs, dim=0)  # [N,1,H,W]
        # 値域統一: [-1,1] なら [0,1] へ
        row_disp = row_batch.to(torch.float32)
        if row_disp.min().item() < 0.0:
            row_disp = (row_disp + 1.0) / 2.0
        row_disp = row_disp.clamp(0.0, 1.0)

        # 個別正規化は使わない（コントラストがバラけるのを防ぐ）
        grid_top = vutils.make_grid(
            row_disp,
            nrow=len(self.top_row_names),
            normalize=True
        )
        self.writer.add_image("00_TopRow/realA_fakeB_realB_fakeA", grid_top, global_step=step)

        # ----- 2) それ以外の各画像を個別にグリッド化して出力 -----
        skip_set = set(self.top_row_names)
        for name, tensor in visuals.items():
            if name in skip_set:
                continue
            if not isinstance(tensor, torch.Tensor):
                continue
            t = self._to_bchw(tensor).to(torch.float32)
            if t.min().item() < 0.0:
                t = (t + 1.0) / 2.0
            t = t.clamp(0.0, 1.0)

            grid = vutils.make_grid(
                t,
                nrow=min(t.shape[0], 8),
                normalize=False
            )
            self.writer.add_image(f"10_AllImages/{name}", grid, global_step=step)

    
    def log_losses(self, losses: Dict[str, float], step: int):
        """損失を TensorBoard に記録"""
        if not isinstance(losses, (dict, OrderedDict)):
            return
        for k, v in losses.items():
            try:
                self.writer.add_scalar(f"Loss/{k}", float(v), global_step=step)
            except Exception:
                # 変換できない場合はスキップ
                continue

    def close(self):
        self.writer.flush()
        self.writer.close()
