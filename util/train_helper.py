# -*- coding: utf8 -*-
"""Training helper utilities for SR-CycleGAN (log reset + visualizers wrapper)."""

import os
import shutil

from util.visualizer import Visualizer as HtmlVisualizer
from util.visualizer2 import Visualizer2 as TBVisualizer


def maybe_reset_logs(opt):
    """
    run_auto.sh から train.py が呼ばれるたびに、
    checkpoints/<SR_CycleGAN系> を “ディレクトリごと”初期化する固定仕様。
    - SR_CycleGAN と SR-CycleGAN の両表記に対応
    - opt.checkpoints_dir 配下に限定する安全チェック付き
    """
    checkpoints_dir = os.path.abspath(opt.checkpoints_dir)
    cand_names = ["SR_CycleGAN", "SR-CycleGAN"]  # 取り違い対策で両方対応

    print(f"[RESET] checkpoints_dir={checkpoints_dir}")
    for nm in cand_names:
        target = os.path.abspath(os.path.join(checkpoints_dir, nm))

        # 安全チェック：checkpoints_dir 配下かつ basename 一致のみ許可
        if not (target + os.sep).startswith(checkpoints_dir + os.sep):
            print(f"[WARN] Skip (outside checkpoints): {target}")
            continue
        if os.path.basename(target) != nm:
            print(f"[WARN] Skip (basename mismatch): {target}")
            continue

        # 削除 → 作り直し
        shutil.rmtree(target, ignore_errors=True)
        os.makedirs(target, exist_ok=True)
        print(f"[INFO] Force-cleaned: {target}")


class TrainLogger:
    """HTML + TensorBoard のラッパ。train.py からの呼び出しをシンプルにする。"""

    def __init__(self, opt):
        self.opt = opt
        self.visualizer_html = HtmlVisualizer(opt)
        self.visualizer_tb = TBVisualizer(opt)

    def log_images(self, model, epoch, total_iters):
        """display_freq ごとに HTML / TensorBoard に画像を書き出す。"""
        if total_iters % self.opt.display_freq != 0:
            return

        save_result = (total_iters % self.opt.update_html_freq == 0)
        model.compute_visuals()
        visuals = model.get_current_visuals()

        # HTML（従来通り）
        self.visualizer_html.display_current_results(visuals, epoch, save_result)
        # TensorBoard（新）
        self.visualizer_tb.display_current_results(visuals, epoch, total_iters)

    def log_losses(self, model, total_iters):
        """
        print_freq ごとに loss を取得して TensorBoard に書き込む。
        戻り値で train.py 側にも返す（tqdm 表示用）。
        """
        if total_iters % self.opt.print_freq != 0:
            return None

        losses = model.get_current_losses()
        self.visualizer_tb.log_losses(losses, total_iters)
        return losses

    def close(self):
        """終了処理。TensorBoard 側のリソースを閉じる。"""
        self.visualizer_tb.close()
