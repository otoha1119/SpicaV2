#!/usr/bin/env bash
# 単枚 DICOM を学習済み G_A (Clinical→Micro) で推論して PNG/DICOM を保存

# 環境: PYTHONPATH に /workspace と /workspace/models を追加（未定義でも安全）
export PYTHONPATH="/workspace:/workspace/models:${PYTHONPATH:-}"

python infer_single_dicom.py \
  --input_dicom /workspace/DataSet/ImageCAS/001.ImgCast/IM_091.dcm \
  --checkpoints_dir /workspace/checkpoints \
  --name SR_CycleGAN \
  --epoch latest \
  --gpu_ids 0 \
  --out_png /workspace/sr.png \
  --out_dicom /workspace/sr.dcm \
  --assume_tanh
