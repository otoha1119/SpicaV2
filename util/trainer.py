# util/trainer.py
# -*- coding: utf8 -*-
"""Trainer class: SR-CycleGAN の学習ループ本体を担当"""

import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image


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
        # サンプル画像保存フラグ（1エポック目1ステップ目のみ）
        self._sample_saved = False

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

                    # ====== 1エポック目1ステップ目のサンプル画像保存 ======
                    if epoch == 1 and i == 0 and not self._sample_saved:
                        self._save_sample_pair(data, epoch, i + 1)

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
    
    def _save_sample_pair(self, data, epoch: int, step: int):
        """1エポック目1ステップ目のペア画像を保存"""
        if self._sample_saved:
            return
        
        # データから画像を取得（バッチの最初のサンプルを使用）
        # DataLoaderが返すデータはバッチ形式なので、最初のサンプルを取得
        if 'lr_crop_np' in data and 'hr_crop_np' in data:
            # numpy配列がテンソルに変換されている場合
            if isinstance(data['lr_crop_np'], torch.Tensor):
                lr_crop = data['lr_crop_np'][0].cpu().numpy()
            else:
                lr_crop = data['lr_crop_np'][0] if isinstance(data['lr_crop_np'], (list, tuple)) else data['lr_crop_np']
            
            if isinstance(data['hr_crop_np'], torch.Tensor):
                hr_crop = data['hr_crop_np'][0].cpu().numpy()
            else:
                hr_crop = data['hr_crop_np'][0] if isinstance(data['hr_crop_np'], (list, tuple)) else data['hr_crop_np']
        else:
            # フォールバック: テンソルから取得（チャンネル次元を除去）
            if isinstance(data['A'], torch.Tensor):
                lr_crop = data['A'][0, 0].cpu().numpy()  # (C, H, W) -> (H, W)
            else:
                lr_crop = data['A'][0, 0] if len(data['A'][0].shape) == 3 else data['A'][0]
            
            if isinstance(data['B'], torch.Tensor):
                hr_crop = data['B'][0, 0].cpu().numpy()  # (C, H, W) -> (H, W)
            else:
                hr_crop = data['B'][0, 0] if len(data['B'][0].shape) == 3 else data['B'][0]
        
        # パス情報を取得
        lr_path = data.get('A_paths', ['unknown'])
        hr_path = data.get('B_paths', ['unknown'])
        if isinstance(lr_path, (list, tuple)):
            lr_path = lr_path[0]
        if isinstance(hr_path, (list, tuple)):
            hr_path = hr_path[0]
        
        # 保存ディレクトリを作成
        save_dir = Path(self.opt.checkpoints_dir) / self.opt.name / "sample_pairs"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像を[0, 1]から[0, 255]に変換
        lr_img_uint8 = (np.clip(lr_crop, 0, 1) * 255).astype(np.uint8)
        hr_img_uint8 = (np.clip(hr_crop, 0, 1) * 255).astype(np.uint8)
        
        # PIL Imageに変換して保存
        lr_pil = Image.fromarray(lr_img_uint8, mode='L')
        hr_pil = Image.fromarray(hr_img_uint8, mode='L')
        
        # ファイル名からパス情報を取得（簡易版）
        lr_name = Path(lr_path).stem if isinstance(lr_path, str) else 'unknown'
        hr_name = Path(hr_path).stem if isinstance(hr_path, str) else 'unknown'
        
        lr_pil.save(save_dir / f"LR_epoch{epoch}_step{step}_{lr_name}.png")
        hr_pil.save(save_dir / f"HR_epoch{epoch}_step{step}_{hr_name}.png")
        
        # サイドバイサイドで保存
        combined = Image.new('L', (lr_crop.shape[1] + hr_crop.shape[1], max(lr_crop.shape[0], hr_crop.shape[0])))
        combined.paste(lr_pil, (0, 0))
        combined.paste(hr_pil, (lr_crop.shape[1], 0))
        combined.save(save_dir / f"pair_epoch{epoch}_step{step}_LR-{lr_name}_HR-{hr_name}.png")
        
        self._sample_saved = True
        print(f"[Trainer] Sample pair saved to {save_dir}")
