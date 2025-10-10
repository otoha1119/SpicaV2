#!/usr/bin/env bash
# MTF pipeline launcher
# - デフォルトはこのファイル内の設定ブロックを使用
# - 必要ならCLI引数で上書き可能（例: --num_rois 300）

set -euo pipefail

### ====================== EDIT HERE (デフォルト設定) ====================== ###
# DICOM ディレクトリ（絶対/相対どちらでもOK）
LR_DIR="/workspace/DataSet/ImageCAS/001.ImgCast"
SR_DIR="/workspace/results/001.ImgCast"
HR_DIR="/workspace/DataSet/photonCT/PhotonCT1024v2/DICOMSAVE-20240514142921-000"



# 出力先
OUT_DIR="/workspace/results"

# 処理パラメータ
NUM_ROIS=150            # 1シリーズあたり抽出するROI数
DRAW_NYQUIST=1          # 1=Nyquist縦線を描画, 0=非表示
USE_CUDA=1              # 1=CUDA使用(可能なら), 0=CPU強制
GPU_IDS="0"             # 例: "0", "0,1", "-1"(CPU扱い)

# SRのPixelSpacing補正倍率（未更新DICOM対策）。空なら補正しない。
SR_SCALE="2.0"          # 例: "2.0" / ""（空文字で無効）

# 再現性
SEED=42
### ====================================================================== ###

usage() {
  echo "Usage: $0 [--lr_dir DIR --sr_dir DIR --hr_dir DIR] [options]"
  echo
  echo "Options (CLIで上書き可):"
  echo "  --lr_dir DIR         LR DICOM dir (default: set in script)"
  echo "  --sr_dir DIR         SR DICOM dir (default: set in script)"
  echo "  --hr_dir DIR         HR DICOM dir (default: set in script)"
  echo "  --out_dir DIR        outputs dir   (default: ${OUT_DIR})"
  echo "  --num_rois N         ROIs per series (default: ${NUM_ROIS})"
  echo "  --draw_nyquist 0|1   (default: ${DRAW_NYQUIST})"
  echo "  --use_cuda 0|1       (default: ${USE_CUDA})"
  echo "  --gpu_ids IDS        e.g. 0 / 0,1 / -1 (default: ${GPU_IDS})"
  echo "  --sr_scale S         e.g. 2.0 ; empty to disable (default: \"${SR_SCALE}\")"
  echo "  --seed N             RNG seed (default: ${SEED})"
  echo "  -h, --help           show this help"
}

# ------------- オプション解析（指定があれば上書き） -------------
while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    --lr_dir)        LR_DIR="$2"; shift; shift ;;
    --sr_dir)        SR_DIR="$2"; shift; shift ;;
    --hr_dir)        HR_DIR="$2"; shift; shift ;;
    --out_dir)       OUT_DIR="$2"; shift; shift ;;
    --num_rois)      NUM_ROIS="$2"; shift; shift ;;
    --draw_nyquist)  DRAW_NYQUIST="$2"; shift; shift ;;
    --use_cuda)      USE_CUDA="$2"; shift; shift ;;
    --gpu_ids)       GPU_IDS="$2"; shift; shift ;;
    --sr_scale)      SR_SCALE="$2"; shift; shift ;;
    --seed)          SEED="$2"; shift; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown option: $key"; usage; exit 1 ;;
  esac
done

# ------------- 前提チェック -------------
if [[ -z "${LR_DIR}" || -z "${SR_DIR}" || -z "${HR_DIR}" ]]; then
  echo "Error: LR_DIR / SR_DIR / HR_DIR が未設定です。スクリプト先頭の設定ブロックを編集するか、CLIで指定してください。" >&2
  usage; exit 1
fi
if [[ ! -d "$LR_DIR" ]]; then echo "Error: LR_DIR not found: $LR_DIR" >&2; exit 1; fi
if [[ ! -d "$SR_DIR" ]]; then echo "Error: SR_DIR not found: $SR_DIR" >&2; exit 1; fi
if [[ ! -d "$HR_DIR" ]]; then echo "Error: HR_DIR not found: $HR_DIR" >&2; exit 1; fi

mkdir -p "$OUT_DIR"

# ------------- GPU可視化設定 -------------
if [[ "${USE_CUDA}" == "1" && "${GPU_IDS}" != "-1" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
else
  # CPU強制（-1 指定時や --use_cuda=0）
  export CUDA_VISIBLE_DEVICES=""
fi

# ------------- 実行 -------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# SR_SCALE が空文字なら main に渡さないための配列トリック
SR_SCALE_ARG=()
if [[ -n "${SR_SCALE}" ]]; then
  SR_SCALE_ARG=(--sr_scale "${SR_SCALE}")
fi

echo "[INFO] LR_DIR=${LR_DIR}"
echo "[INFO] SR_DIR=${SR_DIR}"
echo "[INFO] HR_DIR=${HR_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] NUM_ROIS=${NUM_ROIS}, DRAW_NYQUIST=${DRAW_NYQUIST}"
echo "[INFO] USE_CUDA=${USE_CUDA}, GPU_IDS=${GPU_IDS}, SR_SCALE=${SR_SCALE:-<none>}, SEED=${SEED}"
echo

python3 "${SCRIPT_DIR}/main_mtf.py" \
  --lr_dir "${LR_DIR}" \
  --sr_dir "${SR_DIR}" \
  --hr_dir "${HR_DIR}" \
  --out_dir "${OUT_DIR}" \
  --num_rois "${NUM_ROIS}" \
  --draw_nyquist "${DRAW_NYQUIST}" \
  --use_cuda "${USE_CUDA}" \
  "${SR_SCALE_ARG[@]}" \
  --seed "${SEED}"
