# train.py
# -*- coding: utf8 -*-

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

from util.train_helper import maybe_reset_logs, TrainLogger
from util.trainer import Trainer


if __name__ == '__main__':
    # 1) オプション読み込み（学習設定）
    opt = TrainOptions().parse()

    # 設定表示（scale, SR 倍率, パッチサイズなど）
    print(
        f"[INFO] scale={opt.scale}, sampling_times={opt.sampling_times}, "
        f"lr_patch={opt.lr_patch}, hr_patch={opt.hr_patch}"
    )

    # 2) ログ / checkpoints の初期化
    maybe_reset_logs(opt)

    # 3) データセット & モデル準備
    dataset = create_dataset(opt)

    # インスタンス化
    model = create_model(opt)

    # インスタンスに対してメソッド実行，スケジューラ作成 & チェックポイント読み込み & ネットワーク表示
    model.setup(opt)

    # 4) ロガー & トレーナー準備
    #    - TrainLogger: 画像 & 損失のロギング（HTML + TensorBoard）
    logger = TrainLogger(opt)

    # 学習本体のインスタンス作成
    trainer = Trainer(model, dataset, logger, opt)

    # 5) 学習開始（for epoch / for data … の中身は Trainer 側に隠蔽）
    trainer.train()
