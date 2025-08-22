# -*- coding: utf8 -*-
"""General-purpose training script for SR-CycleGAN (TensorBoard + tqdm, optional clean runs)"""

import os
import shutil
import time
from tqdm import tqdm

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

# HTML を継続して残したい場合（従来の web 出力）
from util.visualizer import Visualizer as HtmlVisualizer
# TensorBoard
from util.visualizer2 import Visualizer2 as TBVisualizer


def maybe_reset_logs(opt):
    """
    実行前にログをリセットしたい場合:
      - 環境変数 RESET_CHECKPOINTS=1 で runs/ を削除
      - 環境変数 RESET_HTML=1        で web/ も削除
    """
    base = os.path.join(opt.checkpoints_dir, opt.name)
    runs_dir = os.path.join(base, "runs")
    web_dir = os.path.join(base, "web")

    if os.environ.get("RESET_CHECKPOINTS", "0") == "1":
        shutil.rmtree(runs_dir, ignore_errors=True)
        os.makedirs(runs_dir, exist_ok=True)
        print(f"[INFO] Cleaned TensorBoard runs: {runs_dir}")

    if os.environ.get("RESET_HTML", "0") == "1":
        shutil.rmtree(web_dir, ignore_errors=True)
        print(f"[INFO] Cleaned HTML dir: {web_dir}")


if __name__ == '__main__':
    # 1) オプション & データセット
    opt = TrainOptions().parse()
    maybe_reset_logs(opt)

    dataset = create_dataset(opt)
    dataset_size = len(dataset)

    # 2) モデル & 可視化
    model = create_model(opt)
    model.setup(opt)

    # 従来 HTML（checkpoints/<name>/web）も残す
    visualizer_html = HtmlVisualizer(opt)
    # TensorBoard
    visualizer_tb = TBVisualizer(opt)

    total_iters = 0  # 累積ステップ（TensorBoard の step）

    # 3) エポックループ
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        with tqdm(total=dataset_size, desc=f"Epoch {epoch}/{opt.niter + opt.niter_decay}", unit="it") as pbar:
            for i, data in enumerate(dataset):
                iter_start_time = time.time()
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                # 学習ステップ
                model.set_input(data)
                model.optimize_parameters(epoch)

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                pbar.update(opt.batch_size)

                # 画像（HTML + TensorBoard）
                if total_iters % opt.display_freq == 0:
                    save_result = (total_iters % opt.update_html_freq == 0)
                    model.compute_visuals()
                    visuals = model.get_current_visuals()

                    # HTML（従来通り）
                    visualizer_html.display_current_results(visuals, epoch, save_result)
                    # TensorBoard（新）
                    visualizer_tb.display_current_results(visuals, epoch, total_iters)

                # 損失（TensorBoard）
                if total_iters % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    # tqdm のステータスにも軽く出す
                    loss_text = " ".join([f"{k}:{float(v):.3f}" for k, v in losses.items()])
                    pbar.set_postfix_str(loss_text, refresh=False)
                    # TensorBoard に記録
                    visualizer_tb.log_losses(losses, total_iters)

                # 途中セーブ
                if total_iters % opt.save_latest_freq == 0:
                    print(f'saving the latest model (epoch {epoch}, total_iters {total_iters})')
                    save_suffix = f'iter_{total_iters}' if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()

        # エポック終端セーブ
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest')
            model.save_networks(epoch)

        print(f'End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {int(time.time() - epoch_start_time)} sec')
        model.update_learning_rate()

    # 終了処理
    visualizer_tb.close()
