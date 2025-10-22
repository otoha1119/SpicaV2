#!/usr/bin/env bash
# MTF pipeline launcher (latest main_mtf.py compatible)
# - このファイル内の設定がデフォルト
# - すべてCLI引数で上書き可能（例: --num_rois 300）

set -euo pipefail

### ====================== EDIT HERE (defaults) ====================== ###
# DICOM dirs
LR_DIR="/workspace/DataSet/ImageCAS/003.ImgCast"
SR_DIR="/workspace/results/001.ImgCast"
HR_DIR="/workspace/DataSet/photonCT/PhotonCT1024v2/DICOMSAVE-20240514142921-000"

# Output
OUT_DIR="/workspace/results"

# ROI extraction / filtering (match main_mtf.py defaults)
NUM_ROIS=4000
ROI_WIDTH=40               # 法線方向の幅（短辺）
ROI_HEIGHT=100             # エッジ方向の高さ（長辺）
ANGLE_MIN=6.0
ANGLE_MAX=12.0
REQUIRE_SINGLE_EDGE=1
EDGE_VERTICAL_TOL_DEG=2.5
MIN_VERTICAL_SPAN_RATIO=0.92
CENTER_TOLERANCE_RATIO=0.08

# Example exports (raw view & ESF/LSF/MTF)
EXPORT_EXAMPLES=1          # 0=off, 1=on
EXAMPLES_PER_SERIES=1

# FFT / axes
DRAW_NYQUIST=0             # 1=draw Nyquist guide
X_NORM=1                   # 1=f/fNyquist, 0=cycles/mm

# Device
USE_CUDA=1                 # 1=CUDA if available, 0=CPU
GPU_IDS="0"                # e.g. "0", "0,1", "-1"(CPU)

# PixelSpacing handling
SR_SCALE="2.0"             # "" to disable SR PixelSpacing adjust
LR_SPACING_ROW="0.3184"    # mm/px; leave empty to skip override
LR_SPACING_COL="0.3184"
HR_SPACING_ROW="0.136719"
HR_SPACING_COL="0.136719"

# ESF reduction (main_mtf: passed through; mtf_core側未対応なら無視)
ESF_FRAC=1.0
ESF_REDUCE="mean"
TRIM_ALPHA=0.1

# Reproducibility
SEED=42
### ================================================================== ###

usage() {
  cat <<EOF
Usage: $0 [--lr_dir DIR --sr_dir DIR --hr_dir DIR] [options]

Options (override defaults):
  --lr_dir DIR
  --sr_dir DIR
  --hr_dir DIR
  --out_dir DIR

  --num_rois N
  --roi_width W
  --roi_height H
  --angle_min DEG
  --angle_max DEG
  --require_single_edge 0|1
  --edge_vertical_tol_deg DEG
  --min_vertical_span_ratio R
  --center_tolerance_ratio R

  --export_examples 0|1
  --examples_per_series N

  --draw_nyquist 0|1
  --x_norm 0|1

  --use_cuda 0|1
  --gpu_ids IDS

  --sr_scale S                 # e.g. 2.0 ; empty to disable
  --lr_spacing ROW COL         # mm/px (two numbers)
  --hr_spacing ROW COL         # mm/px (two numbers)

  --esf_frac R
  --esf_reduce STR             # mean|median など（未対応なら無視）
  --trim_alpha R

  --seed N
  -h, --help
EOF
}

# ---------- parse CLI overrides ----------
while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    --lr_dir)                      LR_DIR="$2"; shift 2 ;;
    --sr_dir)                      SR_DIR="$2"; shift 2 ;;
    --hr_dir)                      HR_DIR="$2"; shift 2 ;;
    --out_dir)                     OUT_DIR="$2"; shift 2 ;;

    --num_rois)                    NUM_ROIS="$2"; shift 2 ;;
    --roi_width)                   ROI_WIDTH="$2"; shift 2 ;;
    --roi_height)                  ROI_HEIGHT="$2"; shift 2 ;;
    --angle_min)                   ANGLE_MIN="$2"; shift 2 ;;
    --angle_max)                   ANGLE_MAX="$2"; shift 2 ;;
    --require_single_edge)         REQUIRE_SINGLE_EDGE="$2"; shift 2 ;;
    --edge_vertical_tol_deg)       EDGE_VERTICAL_TOL_DEG="$2"; shift 2 ;;
    --min_vertical_span_ratio)     MIN_VERTICAL_SPAN_RATIO="$2"; shift 2 ;;
    --center_tolerance_ratio)      CENTER_TOLERANCE_RATIO="$2"; shift 2 ;;

    --export_examples)             EXPORT_EXAMPLES="$2"; shift 2 ;;
    --examples_per_series)         EXAMPLES_PER_SERIES="$2"; shift 2 ;;

    --draw_nyquist)                DRAW_NYQUIST="$2"; shift 2 ;;
    --x_norm)                      X_NORM="$2"; shift 2 ;;

    --use_cuda)                    USE_CUDA="$2"; shift 2 ;;
    --gpu_ids)                     GPU_IDS="$2"; shift 2 ;;

    --sr_scale)                    SR_SCALE="$2"; shift 2 ;;
    --lr_spacing)                  LR_SPACING_ROW="$2"; LR_SPACING_COL="$3"; shift 3 ;;
    --hr_spacing)                  HR_SPACING_ROW="$2"; HR_SPACING_COL="$3"; shift 3 ;;

    --esf_frac)                    ESF_FRAC="$2"; shift 2 ;;
    --esf_reduce)                  ESF_REDUCE="$2"; shift 2 ;;
    --trim_alpha)                  TRIM_ALPHA="$2"; shift 2 ;;

    --seed)                        SEED="$2"; shift 2 ;;
    -h|--help)                     usage; exit 0 ;;
    *) echo "Unknown option: $key"; usage; exit 1 ;;
  esac
done

# ---------- checks ----------
if [[ -z "${LR_DIR}" || -z "${SR_DIR}" || -z "${HR_DIR}" ]]; then
  echo "Error: LR_DIR / SR_DIR / HR_DIR が未設定です。" >&2
  usage; exit 1
fi
[[ -d "$LR_DIR" ]] || { echo "Error: LR_DIR not found: $LR_DIR" >&2; exit 1; }
[[ -d "$SR_DIR" ]] || { echo "Error: SR_DIR not found: $SR_DIR" >&2; exit 1; }
[[ -d "$HR_DIR" ]] || { echo "Error: HR_DIR not found: $HR_DIR" >&2; exit 1; }

mkdir -p "$OUT_DIR"

# ---------- device visibility ----------
if [[ "${USE_CUDA}" == "1" && "${GPU_IDS}" != "-1" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
else
  export CUDA_VISIBLE_DEVICES=""
fi

# ---------- build optional args ----------
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

EXAMPLES_ARGS=(--export_examples "${EXPORT_EXAMPLES}" --examples_per_series "${EXAMPLES_PER_SERIES}")

# ---------- logs ----------
echo "[INFO] LR_DIR=${LR_DIR}"
echo "[INFO] SR_DIR=${SR_DIR}"
echo "[INFO] HR_DIR=${HR_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] NUM_ROIS=${NUM_ROIS}, ROI_WIDTH=${ROI_WIDTH}, ROI_HEIGHT=${ROI_HEIGHT}"
echo "[INFO] ANGLE_MIN=${ANGLE_MIN}, ANGLE_MAX=${ANGLE_MAX}, REQUIRE_SINGLE_EDGE=${REQUIRE_SINGLE_EDGE}"
echo "[INFO] EDGE_VERTICAL_TOL_DEG=${EDGE_VERTICAL_TOL_DEG}, MIN_VERTICAL_SPAN_RATIO=${MIN_VERTICAL_SPAN_RATIO}, CENTER_TOLERANCE_RATIO=${CENTER_TOLERANCE_RATIO}"
echo "[INFO] DRAW_NYQUIST=${DRAW_NYQUIST}, X_NORM=${X_NORM}"
echo "[INFO] USE_CUDA=${USE_CUDA}, GPU_IDS=${GPU_IDS}"
echo "[INFO] SR_SCALE=${SR_SCALE:-<none>}, SEED=${SEED}"
echo "[INFO] LR_SPACING=${LR_SPACING_ROW:-<none>} ${LR_SPACING_COL:-<none>}, HR_SPACING=${HR_SPACING_ROW:-<none>} ${HR_SPACING_COL:-<none>}"
echo "[INFO] ESF_FRAC=${ESF_FRAC}, ESF_REDUCE=${ESF_REDUCE}, TRIM_ALPHA=${TRIM_ALPHA}"
echo

# ---------- run ----------
python3 "${SCRIPT_DIR}/main_mtf.py" \
  --lr_dir "${LR_DIR}" \
  --sr_dir "${SR_DIR}" \
  --hr_dir "${HR_DIR}" \
  --out_dir "${OUT_DIR}" \
  --num_rois "${NUM_ROIS}" \
  --roi_width "${ROI_WIDTH}" \
  --roi_height "${ROI_HEIGHT}" \
  --angle_min "${ANGLE_MIN}" \
  --angle_max "${ANGLE_MAX}" \
  --require_single_edge "${REQUIRE_SINGLE_EDGE}" \
  --edge_vertical_tol_deg "${EDGE_VERTICAL_TOL_DEG}" \
  --min_vertical_span_ratio "${MIN_VERTICAL_SPAN_RATIO}" \
  --center_tolerance_ratio "${CENTER_TOLERANCE_RATIO}" \
  --draw_nyquist "${DRAW_NYQUIST}" \
  --x_norm "${X_NORM}" \
  --use_cuda "${USE_CUDA}" \
  --gpu_ids "${GPU_IDS}" \
  --esf_frac "${ESF_FRAC}" \
  --esf_reduce "${ESF_REDUCE}" \
  --trim_alpha "${TRIM_ALPHA}" \
  "${EXAMPLES_ARGS[@]}" \
  "${SR_SCALE_ARG[@]}" \
  "${LR_SPACING_ARG[@]}" \
  "${HR_SPACING_ARG[@]}" \
  --seed "${SEED}"
