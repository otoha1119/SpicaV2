#!/bin/bash
#実行コード
"""
chmod +x run_auto.sh
./run_auto.sh   
"""
# ==== 環境設定 ====
# GPUを指定（0番GPUを使用）
export CUDA_VISIBLE_DEVICES=0

# 実験名（結果保存用フォルダに使われる）
EXP_NAME="exp_sr2x"

# データパス
LR_ROOT="/workspace/DataSet/ImageCAS"
HR_ROOT="/workspace/DataSet/photonCT/PhotonCT1024v2"

# ==== 学習 ====
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

# # ==== 推論 ====
# # 推論対象DICOMファイル（512×512のLR CT画像を指定）
# INPUT_DICOM="/workspace/DataSet/test_input/sample_512.dcm"
# OUTPUT_DICOM="./results/SR_2x.dcm"

# echo "=== 推論を開始します ==="
# python medicaltest_dicom.py \
#   --name ${EXP_NAME} \
#   --model medical_cycle_gan \
#   --dataset_mode dicom_ctpcct_2x_test \
#   --input_dicom ${INPUT_DICOM} \
#   --epoch latest \
#   --gpu_ids 0 \
#   --halves_pixel_spacing \
#   --output_dicom ${OUTPUT_DICOM}

echo "=== 完了しました！ 出力: ${OUTPUT_DICOM} ==="
