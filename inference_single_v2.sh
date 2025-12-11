#!/usr/bin/env bash
set -euo pipefail

# Adjustable parameters (override via environment variables)
INPUT_DICOM=${INPUT_DICOM:-/workspace/IM_091.dcm}
OUTPUT_DICOM=${OUTPUT_DICOM:-/workspace/results/SR_2x_v2.dcm}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-/workspace/checkpoints_mac}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-SR_CycleGAN}
EPOCH=${EPOCH:-200}
GPU_IDS=${GPU_IDS:-0}
USE_G=${USE_G:-A}

python inference_single_v2.py \
  --dataset_mode dicom_ctpcct_2x_test \
  --model medical_cycle_gan \
  --clinical2micronetG clinical_to_micro_resnet_9blocks \
  --micro2clinicalnetG micro_to_clinical_resnet_9blocks \
  --netG resnet_9blocks \
  --ngf 64 \
  --input_nc 1 --output_nc 1 \
  --name "${EXPERIMENT_NAME}" \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --epoch "${EPOCH}" \
  --use_G "${USE_G}" \
  --input_dicom "${INPUT_DICOM}" \
  --output_dicom "${OUTPUT_DICOM}" \
  --gpu_ids "${GPU_IDS}" \
  --sampling_times 1
