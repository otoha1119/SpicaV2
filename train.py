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
    base = os.path.join(opt.checkpoints_dir, opt.name)
    runs_dir = os.path.join(base, "runs")
    web_dir  = os.path.join(base, "web")

    # CLI 優先、無ければ環境変数
    reset_all    = getattr(opt, 'reset_all', False)    or os.environ.get("RESET_ALL", "0") == "1"
    reset_runs   = getattr(opt, 'reset_runs', False)   or os.environ.get("RESET_RUNS", "0") == "1"
    reset_html   = getattr(opt, 'reset_html', False)   or os.environ.get("RESET_HTML", "0") == "1"
    reset_models = getattr(opt, 'reset_models', False) or os.environ.get("RESET_MODELS", "0") == "1"

    if reset_all:
        shutil.rmtree(base, ignore_errors=True)
        os.makedirs(base, exist_ok=True)
        print(f"[INFO] Cleaned ALL: {base}")
        return

    if reset_runs:
        shutil.rmtree(runs_dir, ignore_errors=True)
        os.makedirs(runs_dir, exist_ok=True)
        print(f"[INFO] Cleaned TensorBoard runs: {runs_dir}")

    if reset_html:
        shutil.rmtree(web_dir, ignore_errors=True)
        print(f"[INFO] Cleaned HTML dir: {web_dir}")

    if reset_models:
        # *.pth や iter_* / latest などを削除（必要に応じてパターン追加）
        from glob import glob
        patterns = [
            os.path.join(base, "*.pth"),
            os.path.join(base, "*.pt"),
            os.path.join(base, "iter_*"),
            os.path.join(base, "latest*"),
            os.path.join(base, "epoch_*"),
            os.path.join(base, "loss_log.txt"),
            os.path.join(base, "events.out.tfevents*"),
        ]
        for pat in patterns:
            for p in glob(pat):
                try:
                    shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
                except Exception:
                    pass
        print(f"[INFO] Cleaned model checkpoints & logs under: {base}")

if __name__ == '__main__':
    # 1) オプション & データセット
    opt = TrainOptions().parse() #オプションを取得

    print(f"[INFO] scale={opt.scale}, sampling_times={opt.sampling_times}, "
      f"lr_patch={opt.lr_patch}, hr_patch={opt.hr_patch}")


    maybe_reset_logs(opt) #上記関数にてoptionログを削除

    dataset = create_dataset(opt) #データセット作成
    dataset_size = len(dataset) #1epochあたりのデータ枚数

    # 2) モデル & 可視化
    model = create_model(opt) #モデルの生成(models/__init__.py)
    model.setup(opt)          #継承元のbase_model.pyのsetup()を呼び出し(学習率スケジューラのセット，モデルのロード，ネットワーク構造の表示)

    # 従来 HTML（checkpoints/<name>/web）も残す
    visualizer_html = HtmlVisualizer(opt)

    # TensorBoard
    visualizer_tb = TBVisualizer(opt)

    total_iters = 0  # 累積ステップ（TensorBoard の step）

    # 3) エポックループ
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time() #Epochの開始時間の記録
        iter_data_time = time.time() #データ読み込み開始時刻の記録
        epoch_iter = 0 #イテレーションカウンタ

        with tqdm(total=dataset_size, desc=f"Epoch {epoch}/{opt.niter + opt.niter_decay}", unit="it") as pbar: #tqdmのプログレスバー
            for i, data in enumerate(dataset): #データ取り出し(データローダ)
                iter_start_time = time.time() #イテレーション開始時刻の記録
                if total_iters % opt.print_freq == 0: 
                    t_data = iter_start_time - iter_data_time #イテレーション間の時間計測

                # 学習ステップ
                model.set_input(data) #dataをモデル対応に変換,バッチをGPUに搭載
                model.optimize_parameters(epoch) #学習本体

                total_iters += opt.batch_size #累積イテレーションの更新
                epoch_iter += opt.batch_size #エポック内イテレーションの更新
                pbar.update(opt.batch_size) #tqdmのプログレスバー更新

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
                    #pbar.set_postfix_str(loss_text, refresh=False)
                    # TensorBoard に記録
                    visualizer_tb.log_losses(losses, total_iters)

                # 途中セーブ
                if total_iters % opt.save_latest_freq == 0:
                    #print(f'saving the latest model (epoch {epoch}, total_iters {total_iters})')
                    save_suffix = f'iter_{total_iters}' if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time() #データ読み込み開始時刻の更新

        # エポック終端セーブ
        if epoch % opt.save_epoch_freq == 0: #何エポックごとに保存か
            model.save_networks('latest')
            model.save_networks(epoch)

        print(f'End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {int(time.time() - epoch_start_time)} sec')
        model.update_learning_rate() #学習率更新

    # 終了処理
    visualizer_tb.close()
