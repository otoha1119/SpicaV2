#!/usr/bin/env bash
set -euo pipefail

# DICOM ディレクトリ
LR_DIR="/workspace/DataSet/ImageCAS/003.ImgCast"
SR_DIR="/workspace/results/001.ImgCast"
HR_DIR="/workspace/DataSet/photonCT/PhotonCT1024v2/DICOMSAVE-20240514142921-000"

# 出力先
OUT_DIR="/workspace/results"

# ---- mtf_v2.py ----
NUM_ROIS=4000
DRAW_NYQUIST=0
USE_CUDA=1
GPU_IDS="0"
SR_SCALE="2.0"
X_NORM=1
LR_SPACING_ROW="0.3184"
LR_SPACING_COL="0.3184"
HR_SPACING_ROW="0.136719"
HR_SPACING_COL="0.136719"
SEED=42

# 心臓クロップ（1-basedスライス）
HR_Z_START=1; HR_Z_END=550
HR_X1=90; HR_Y1=190; HR_X2=850; HR_Y2=830
SR_Z_START=1; SR_Z_END=210
SR_X1=60; SR_Y1=124; SR_X2=800; SR_Y2=710
LR_Z_START=1; LR_Z_END=210
LR_X1=30; LR_Y1=62; LR_X2=400; LR_Y2=355

if [[ ! -d "$LR_DIR" ]]; then echo "Error: LR_DIR not found: $LR_DIR" >&2; exit 1; fi
if [[ ! -d "$SR_DIR" ]]; then echo "Error: SR_DIR not found: $SR_DIR" >&2; exit 1; fi
if [[ ! -d "$HR_DIR" ]]; then echo "Error: HR_DIR not found: $HR_DIR" >&2; exit 1; fi

mkdir -p "$OUT_DIR"

if [[ "${USE_CUDA}" == "1" && "${GPU_IDS}" != "-1" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
else
  export CUDA_VISIBLE_DEVICES=""
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SR_SCALE_ARG=()
[[ -n "${SR_SCALE}" ]] && SR_SCALE_ARG=(--sr_scale "${SR_SCALE}")

LR_SPACING_ARG=()
if [[ -n "${LR_SPACING_ROW}" && -n "${LR_SPACING_COL}" ]]; then
  LR_SPACING_ARG=(--lr_spacing "${LR_SPACING_ROW}" "${LR_SPACING_COL}")
fi

HR_SPACING_ARG=()
if [[ -n "${HR_SPACING_ROW}" && -n "${HR_SPACING_COL}" ]]; then
  HR_SPACING_ARG=(--hr_spacing "${HR_SPACING_ROW}" "${HR_SPACING_COL}")
fi

python3 "${SCRIPT_DIR}/mtf_v2.py" \
  --lr_dir "${LR_DIR}" \
  --sr_dir "${SR_DIR}" \
  --hr_dir "${HR_DIR}" \
  --out_dir "${OUT_DIR}" \
  --num_rois "${NUM_ROIS}" \
  --draw_nyquist "${DRAW_NYQUIST}" \
  --use_cuda "${USE_CUDA}" \
  --gpu_ids "${GPU_IDS}" \
  "${SR_SCALE_ARG[@]}" \
  "${LR_SPACING_ARG[@]}" \
  "${HR_SPACING_ARG[@]}" \
  --x_norm "${X_NORM}" \
  --seed "${SEED}" \
  --hr_z "${HR_Z_START}" "${HR_Z_END}" \
  --hr_xyxy "${HR_X1}" "${HR_Y1}" "${HR_X2}" "${HR_Y2}" \
  --sr_z "${SR_Z_START}" "${SR_Z_END}" \
  --sr_xyxy "${SR_X1}" "${SR_Y1}" "${SR_X2}" "${SR_Y2}" \
  --lr_z "${LR_Z_START}" "${LR_Z_END}" \
  --lr_xyxy "${LR_X1}" "${LR_Y1}" "${LR_X2}" "${LR_Y2}"
