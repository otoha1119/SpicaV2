#!/bin/bash
set -e

echo "=== 依存関係をインストールします ==="
apt-get update && apt-get install -y libgl1-mesa-glx
pip install -r requirements.txt
echo "=== 依存関係インストール完了 ==="

# データパス指定
LR_ROOT="/workspace/DataSet/ImageCAS"
HR_ROOT="/workspace/DataSet/photonCT/PhotonCT1024v2"
EXP_NAME="SR_CycleGAN"

# 学習開始
echo "=== 学習を開始します ==="
python train.py \
  --name ${EXP_NAME} \
  --model medical_cycle_gan \
  --dataset_mode dicom_ctpcct_2x \
  --gpu_ids 0 \
  --lr_root ${LR_ROOT} \
  --hr_root ${HR_ROOT} \
  --lr_patch 98 \
  --hr_patch 196 \
  --hr_oversample_ratio 1.0 \
  --batch_size 1 \
  --sampling_times 1

echo "=== 完了しました！ 出力:  ==="
