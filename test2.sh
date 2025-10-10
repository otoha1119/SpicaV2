#!/usr/bin/env bash
set -euo pipefail

IN_DIR="/workspace/DataSet/ImageCAS/001.ImgCast"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="/workspace/results/001.ImgCast"
GPU_IDS="0"

export INPUT_DIR="$IN_DIR"
export OUTPUT_DIR="$OUT_DIR"

mkdir -p "${OUT_DIR}"

python inference_multi.py \
  --dataset_mode dicom_ctpcct_2x_test \
  --model medical_cycle_gan \
  --clinical2micronetG clinical_to_micro_resnet_9blocks \
  --micro2clinicalnetG micro_to_clinical_resnet_9blocks \
  --netG resnet_9blocks \
  --ngf 64 \
  --input_nc 1 --output_nc 1 \
  --name SR_CycleGAN \
  --checkpoints_dir /workspace/checkpoints_mac \
  --epoch 167 \
  --use_G A \
  --halves_pixel_spacing \
  --pad_mod 4 \
  --gpu_ids "${GPU_IDS}" \
  --sampling_times 1

echo "[DONE] Results saved under: ${OUT_DIR}"
