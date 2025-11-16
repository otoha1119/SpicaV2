# util/train_helper.py
# -*- coding: utf8 -*-
"""Training helper utilities for SR-CycleGAN (log reset + visualizers wrapper)."""

import os
import shutil

# HTML 可視化用の Visualizer
from util.visualizer import Visualizer as HtmlVisualizer
# TensorBoard 用の Visualizer
from util.visualizer2 import Visualizer2 as TBVisualizer


def maybe_reset_logs(opt):
    """
    checkpoints ディレクトリを「毎回きれいな状態」にするための関数。
    - SR_CycleGAN / SR-CycleGAN という 2 種類のフォルダ名に対応
    - opt.checkpoints_dir 配下だけを安全に削除＆再作成する
    """
    # 絶対パスに揃えておく（安全のため）
    checkpoints_dir = os.path.abspath(opt.checkpoints_dir)

    # 想定される実験名（フォルダ名）の候補
    cand_names = ["SR_CycleGAN", "SR-CycleGAN"]

    print(f"[RESET] checkpoints_dir={checkpoints_dir}")
    for nm in cand_names:
        # checkpoints_dir / nm というパスを作る
        target = os.path.abspath(os.path.join(checkpoints_dir, nm))

        # --- 安全チェック その1: checkpoints_dir 配下か？ ---
        if not (target + os.sep).startswith(checkpoints_dir + os.sep):
            print(f"[WARN] Skip (outside checkpoints): {target}")
            continue

        # --- 安全チェック その2: basename が想定名と一致しているか？ ---
        if os.path.basename(target) != nm:
            print(f"[WARN] Skip (basename mismatch): {target}")
            continue

        # 既存ディレクトリを削除（あってもなくても OK）
        shutil.rmtree(target, ignore_errors=True)
        # 空ディレクトリとして作り直す
        os.makedirs(target, exist_ok=True)
        print(f"[INFO] Force-cleaned: {target}")


class TrainLogger:
    """
    HTML + TensorBoard をまとめて扱うロガークラス。
    train.py / Trainer からは、このクラスだけ意識すれば OK にする。
    """

    def __init__(self, opt):
        # オプションを保持しておく（display_freq など見るため）
        self.opt = opt
        # 画像を HTML で保存する Visualizer（従来のもの）
        self.visualizer_html = HtmlVisualizer(opt)
        # TensorBoard に書き込む Visualizer（新）
        self.visualizer_tb = TBVisualizer(opt)

    def log_images(self, model, epoch, total_iters):
        """
        画像のログ出力（HTML + TensorBoard）。
        display_freq ごとに実行される。
        """
        # 指定ステップでなければ何もしない
        if total_iters % self.opt.display_freq != 0:
            return

        # 一定間隔で HTML を保存するかどうか（毎回は重いので）
        save_result = (total_iters % self.opt.update_html_freq == 0)

        # モデル内部で可視化用の画像を計算（必要なら）
        model.compute_visuals()
        # 現在の可視化用画像を取得（LR, SR, HR など）
        visuals = model.get_current_visuals()

        # HTML 側に書き出し
        self.visualizer_html.display_current_results(visuals, epoch, save_result)
        # TensorBoard 側にも書き出し
        self.visualizer_tb.display_current_results(visuals, epoch, total_iters)

    def log_losses(self, model, total_iters):
        """
        損失のログ出力（TensorBoard）。
        print_freq ごとに実行される。
        戻り値として losses を返しておくと、tqdm 側でも使える。
        """
        # 指定ステップでなければ何もしない
        if total_iters % self.opt.print_freq != 0:
            return None

        # モデルから現在の loss 値をまとめて取得
        losses = model.get_current_losses()
        # TensorBoard に書き込み
        self.visualizer_tb.log_losses(losses, total_iters)
        return losses

    def close(self):
        """
        終了処理。
        TensorBoard 用の writer などを閉じる（必要に応じて）。
        """
        self.visualizer_tb.close()
