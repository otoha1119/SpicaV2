# util/trainer.py
# -*- coding: utf8 -*-
"""Trainer class: SR-CycleGAN の学習ループ本体を担当"""

import time
from tqdm import tqdm


class Trainer:
    """
    モデル・データセット・ロガーを受け取って、
    「エポックループ ＋ イテレーションループ」を回すクラス。
    """

    def __init__(self, model, dataset, logger, opt):
        """
        Args:
            model: MedicalCycleGANModel のインスタンス
            dataset: create_dataset(opt) で作られたデータセット
            logger: TrainLogger インスタンス（HTML + TensorBoard）
            opt: TrainOptions（学習設定）
        """
        self.model = model
        self.dataset = dataset
        self.logger = logger
        self.opt = opt

        # 1epoch あたりのサンプル数
        self.dataset_size = len(dataset)
        # これまでに処理した「累積サンプル数」（TensorBoard の step にも使う）
        self.total_iters = 0

    def train(self):
        """学習ループ本体（エポックループ + イテレーションループ）"""
        opt = self.opt
        model = self.model
        logger = self.logger
        dataset = self.dataset

        # epoch_count 〜 niter + niter_decay まで回す（CycleGAN 本家仕様）
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()  # エポック開始時間
            iter_data_time = time.time()    # データ読み込み開始時間
            epoch_iter = 0                  # このエポック内で処理したサンプル数

            # tqdm のプログレスバー（1epoch 分）
            with tqdm(
                total=self.dataset_size,
                desc=f"Epoch {epoch}/{opt.niter + opt.niter_decay}",
                unit="it",
            ) as pbar:

                # DataLoader から (画像バッチ + メタ情報) を取得
                for i, data in enumerate(dataset):
                    iter_start_time = time.time()
                    # print_freq ごとに「データ読み込み時間」を計測したければここを使う
                    if self.total_iters % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time  # 今は使っていないが残しておく

                    # ====== 1 ステップ分の学習処理 ======

                    # 1) 入力を GPU に載せて、model 内の self.real_A / self.real_B をセット
                    model.set_input(data)

                    # 2) Generator・Discriminator の loss.backward() + optimizer.step()
                    model.optimize_parameters(epoch)

                    # 累積サンプル数と、このエポック内カウンタを更新
                    self.total_iters += opt.batch_size
                    epoch_iter += opt.batch_size
                    # プログレスバーも進める
                    pbar.update(opt.batch_size)

                    # ====== 画像のログ（HTML + TensorBoard） ======
                    logger.log_images(model, epoch, self.total_iters)

                    # ====== 損失のログ（TensorBoard + tqdm 用文字列） ======
                    losses = logger.log_losses(model, self.total_iters)
                    if losses is not None:
                        # "G:0.123 D_A:0.456 ..." みたいな表示用文字列
                        loss_text = " ".join(
                            [f"{k}:{float(v):.3f}" for k, v in losses.items()]
                        )
                        # tqdm のステータスに載せたければ以下を有効化
                        # pbar.set_postfix_str(loss_text, refresh=False)

                    # ====== 途中セーブ ======
                    if self.total_iters % opt.save_latest_freq == 0:
                        # イテレーション番号で保存するか / latest で上書きするか
                        save_suffix = (
                            f'iter_{self.total_iters}'
                            if opt.save_by_iter
                            else 'latest'
                        )
                        model.save_networks(save_suffix)

                    # 次のループ用に「データ読み込み開始時間」を更新
                    iter_data_time = time.time()

            # ====== エポック終端でのセーブ ======
            # 指定エポックごとに「latest」と「epoch 番号」で保存
            if epoch % opt.save_epoch_freq == 0:
                model.save_networks('latest')
                model.save_networks(epoch)

            # エポックの処理時間を表示
            print(
                f'End of epoch {epoch} / {opt.niter + opt.niter_decay} '
                f'\t Time Taken: {int(time.time() - epoch_start_time)} sec'
            )

            # ====== 学習率更新（スケジューラ） ======
            model.update_learning_rate()

        # ====== すべての学習終了後の後処理 ======
        logger.close()
