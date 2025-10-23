# -*- coding: utf8 -*-
"""TensorBoard visualizer (Visualizer2) — minimal & focused (v2, backward‑compatible)

変更点（v2）:
- train.py から `display_current_results(visuals, epoch, total_iters)` と
  呼ばれても動くように、可変引数に対応しました。
- step は「与えられた引数の最後の整数」を採用（なければ 0）。

基本仕様:
- TensorBoard に **最新バッチの SR / LR / HR の3枚だけ** を記録します。
- それ以外の画像は書き出しません。
- キー: LR=real_A, SR=fake_B, HR=real_B（順序は self.top_row_names で調整可）
"""

import os
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


class Visualizer2:
    """TensorBoard Visualizer (SR/LR/HR の3枚のみ出力)"""

    def __init__(self, opt):
        self.opt = opt
        self.log_dir = os.path.join(getattr(opt, "checkpoints_dir", "./checkpoints"),
                                    getattr(opt, "name", "default"),
                                    "runs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # 表示順（左→右）。デフォは SR, LR, HR
        self.top_row_names: List[str] = ["fake_B", "real_A", "real_B"]

        self.grid_padding: int = 2
        self.target_edge: int = int(getattr(opt, "display_winsize", 256) or 256)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -------------------------- helpers --------------------------

    def _to_bchw(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return None
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        if t.dim() == 2:             # [H, W]
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:           # [C, H, W] or [H, W, C]
            if t.shape[0] in (1, 3): # [C, H, W]
                t = t.unsqueeze(0)
            else:                    # [H, W, C]
                t = t.permute(2, 0, 1).unsqueeze(0)
        elif t.dim() == 4:
            pass
        else:
            raise ValueError(f"Unsupported tensor shape: {t.shape}")
        return t

    def _normalize01(self, t: torch.Tensor) -> torch.Tensor:
        if t.min() < -0.5 and t.max() <= 1.5:
            t = (t + 1.0) * 0.5
        tmin = float(t.min())
        tmax = float(t.max())
        if tmin == tmax:
            return torch.zeros_like(t)
        t = (t - tmin) / (tmax - tmin)
        return t.clamp(0, 1)

    def _take_latest(self, t: torch.Tensor) -> torch.Tensor:
        if t is None:
            return None
        if t.dim() == 4 and t.shape[0] > 1:
            return t[-1:].contiguous()
        return t

    def _resize_to_edge(self, t: torch.Tensor, target_edge: int) -> torch.Tensor:
        if t is None:
            return None
        _, _, h, w = t.shape
        if h == 0 or w == 0:
            return t
        long_edge = max(h, w)
        if long_edge == target_edge:
            return t
        scale = target_edge / float(long_edge)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        return F.interpolate(t, size=(new_h, new_w), mode="nearest")

    def _pad_to_same_hw(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        if not tensors:
            return tensors
        max_h = max(t.shape[2] for t in tensors if t is not None)
        max_w = max(t.shape[3] for t in tensors if t is not None)
        padded = []
        for t in tensors:
            if t is None:
                continue
            _, _, h, w = t.shape
            pad_h = max_h - h
            pad_w = max_w - w
            t = F.pad(t, (0, pad_w, 0, pad_h))
            padded.append(t)
        return padded

    # --------------------------- public ---------------------------

    @torch.no_grad()
    def display_current_results(self, visuals: Dict[str, Any], *args, **kwargs):
        """最新バッチの SR/LR/HR の3枚だけを TensorBoard に出力

        互換仕様:
        - display_current_results(visuals, step)
        - display_current_results(visuals, epoch, total_iters)
        - display_current_results(visuals, ..., step=<int>)
        など、最後に渡された整数を global_step として使います。
        """
        # global_step を決定
        step = kwargs.get("step", None)
        if step is None:
            # 可変長引数の最後の int を採用
            for x in reversed(args):
                if isinstance(x, int):
                    step = x
                    break
        if step is None:
            step = 0  # フォールバック

        if not isinstance(visuals, dict):
            return

        row_imgs: List[torch.Tensor] = []
        for name in self.top_row_names:
            img = visuals.get(name, None)
            if not isinstance(img, torch.Tensor):
                row_imgs.append(torch.zeros(1, 1, self.target_edge, self.target_edge))
                continue
            t = self._to_bchw(img)
            t = self._take_latest(t)
            t = self._normalize01(t)
            t = self._resize_to_edge(t, self.target_edge)
            row_imgs.append(t.cpu())

        row_imgs = self._pad_to_same_hw(row_imgs)

        if len(row_imgs) > 0:
            grid = vutils.make_grid(torch.cat(row_imgs, dim=0), nrow=len(row_imgs),
                                    padding=self.grid_padding, normalize=False)
            self.writer.add_image("00_TopRow/SR_LR_HR", grid, global_step=step)

    def log_losses(self, losses: Dict[str, float], *args, **kwargs):
        """損失を TensorBoard に記録（引数互換: 最後の int を step として使用）"""
        step = kwargs.get("step", None)
        if step is None:
            for x in reversed(args):
                if isinstance(x, int):
                    step = x
                    break
        if step is None:
            step = 0
        if not isinstance(losses, dict):
            return
        for k, v in losses.items():
            try:
                self.writer.add_scalar(f"Loss/{k}", float(v), global_step=step)
            except Exception:
                continue

    def close(self):
        self.writer.flush()
        self.writer.close()
