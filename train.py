# -*- coding: utf8 -*-
"""General-purpose training script for SR-CycleGAN (TensorBoard + tqdm, optional clean runs)"""

import os
import time
from tqdm import tqdm

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

# 新しく作ったヘルパ
from util.train_helper import maybe_reset_logs, TrainLogger


if __name__ == '__main__':
    # 1) オプション & データセット
    opt = TrainOptions().parse()  # オプションを取得

    print(
        f"[INFO] scale={opt.scale}, sampling_times={opt.sampling_times}, "
        f"lr_patch={opt.lr_patch}, hr_patch={opt.hr_patch}"
    )

    # ログ／checkpoints のクリーンアップ
    maybe_reset_logs(opt)

    # データセット作成
    dataset = create_dataset(opt)
    dataset_size = len(dataset)  # 1epochあたりのデータ枚数

    # 2) モデル & ロガー
    model = create_model(opt)  # モデルの生成(models/__init__.py)
    model.setup(opt)           # base_model.py の setup() を呼び出し

    logger = TrainLogger(opt)  # HTML + TensorBoard ラッパ
    total_iters = 0            # 累積ステップ（TensorBoard の step）

    # 3) エポックループ
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()   # Epoch の開始時間
        iter_data_time = time.time()     # データ読み込み開始時刻
        epoch_iter = 0                   # エポック内イテレーションカウンタ

        with tqdm(
            total=dataset_size,
            desc=f"Epoch {epoch}/{opt.niter + opt.niter_decay}",
            unit="it",
        ) as pbar:  # tqdm のプログレスバー

            for i, data in enumerate(dataset):  # データローダ
                iter_start_time = time.time()
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time  # データ読み込み時間（必要なら使う）

                # 学習ステップ
                model.set_input(data)          # data をモデル入力に変換 + GPU 搭載
                model.optimize_parameters(epoch)  # 学習本体

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                pbar.update(opt.batch_size)    # tqdm のプログレスバー更新

                # 画像（HTML + TensorBoard）
                logger.log_images(model, epoch, total_iters)

                # 損失（TensorBoard + tqdm）
                losses = logger.log_losses(model, total_iters)
                if losses is not None:
                    loss_text = " ".join(
                        [f"{k}:{float(v):.3f}" for k, v in losses.items()]
                    )
                    # tqdm のステータスに出したい場合はコメントアウト外す
                    # pbar.set_postfix_str(loss_text, refresh=False)

                # 途中セーブ
                if total_iters % opt.save_latest_freq == 0:
                    # print(f'saving the latest model (epoch {epoch}, total_iters {total_iters})')
                    save_suffix = f'iter_{total_iters}' if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()  # データ読み込み開始時刻の更新

        # エポック終端セーブ
        if epoch % opt.save_epoch_freq == 0:  # 何エポックごとに保存か
            model.save_networks('latest')
            model.save_networks(epoch)

        print(
            f'End of epoch {epoch} / {opt.niter + opt.niter_decay} '
            f'\t Time Taken: {int(time.time() - epoch_start_time)} sec'
        )
        model.update_learning_rate()  # 学習率更新

    # 終了処理
    logger.close()
