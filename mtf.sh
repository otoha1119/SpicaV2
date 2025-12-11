#!/usr/bin/env bash
# MTF pipeline launcher (single-switch version)

<< COMMENTOUT
chmod +x mtf.sh 
./mtf.sh
COMMENTOUT

set -euo pipefail

### ================== 設定（ここだけ編集すればOK） ================== ###
# DICOM ディレクトリ
LR_DIR="/workspace/DataSet/ImageCAS_v3/EID-003"
SR_DIR="/workspace/results/EID-003-results"
HR_DIR="/workspace/DataSet/photonCT/PhotonCT1024v3/PCD-001"

# 出力先
OUT_DIR="/workspace/results"

# ---- 通常モード用（main_mtf.py） ----
NUM_ROIS=4000         # 1シリーズあたりのROI数
DRAW_NYQUIST=0        # 1=Nyquist線描画, 0=非表示
USE_CUDA=0            # 1=CUDA使用, 0=CPU
GPU_IDS="0"           # "0" / "0,1" / "-1"(CPU)
SR_SCALE="2.0"        # SRのPixelSpacing補正倍率。空文字""で無効
X_NORM=1              # 1=正規化(主観比較向け), 0=物理単位[cycles/mm]
LR_SPACING_ROW="0.3184"   # 空ならDICOMの PixelSpacing を使用
LR_SPACING_COL="0.3184"
HR_SPACING_ROW="0.136719"
HR_SPACING_COL="0.136719"
SEED=42

# ---- サンプル可視化（viz_sample） ----
VIZ_SAMPLE=0     # 1=サンプル可視化モード / 0=通常モード
SERIES_ID="sample"    # 出力サブディレクトリ名 (outputs/sample_viz/<SERIES_ID>/...)
SAMPLE_INDEX=0        # 可視化するROIインデックス（0起点）
### ================================================================ ###

usage() {
  echo "Usage: $0"
  echo "※ 引数不要。スイッチはファイル先頭の変数で切り替え。"
  echo
  echo "主なスイッチ:"
  echo "  VIZ_SAMPLE=1/0   -> サンプル可視化 ON/OFF"
  echo "  X_NORM=1/0       -> x軸 正規化 ON/OFF（通常モードのみ）"
}

# ------------- 前提チェック -------------
if [[ -z "${LR_DIR}" || -z "${SR_DIR}" || -z "${HR_DIR}" ]]; then
  echo "Error: LR_DIR / SR_DIR / HR_DIR が未設定です。" >&2
  usage; exit 1
fi
if [[ ! -d "$LR_DIR" ]]; then echo "Error: LR_DIR not found: $LR_DIR" >&2; exit 1; fi
if [[ ! -d "$SR_DIR" ]]; then echo "Error: SR_DIR not found: $SR_DIR" >&2; exit 1; fi
if [[ ! -d "$HR_DIR" ]]; then echo "Error: HR_DIR not found: $HR_DIR" >&2; exit 1; fi

mkdir -p "$OUT_DIR"

# ------------- GPU設定 -------------
if [[ "${USE_CUDA}" == "1" && "${GPU_IDS}" != "-1" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
else
  export CUDA_VISIBLE_DEVICES=""
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# オプション引数を配列化（空は渡さない）
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

X_NORM_ARG=(--x_norm "${X_NORM}")

echo "[INFO] LR_DIR=${LR_DIR}"
echo "[INFO] SR_DIR=${SR_DIR}"
echo "[INFO] HR_DIR=${HR_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] NUM_ROIS=${NUM_ROIS}, DRAW_NYQUIST=${DRAW_NYQUIST}"
echo "[INFO] USE_CUDA=${USE_CUDA}, GPU_IDS=${GPU_IDS}"
echo "[INFO] SR_SCALE=${SR_SCALE:-<none>}, X_NORM=${X_NORM}, SEED=${SEED}"
echo "[INFO] LR_SPACING=${LR_SPACING_ROW:-<none>} ${LR_SPACING_COL:-<none>}, HR_SPACING=${HR_SPACING_ROW:-<none>} ${HR_SPACING_COL:-<none>}"
echo "[INFO] VIZ_SAMPLE=${VIZ_SAMPLE}, SERIES_ID=${SERIES_ID}, SAMPLE_INDEX=${SAMPLE_INDEX}"
echo

if [[ "${VIZ_SAMPLE}" == "1" ]]; then
  # ------ サンプル可視化モード ------
  # 相対インポート(.utils 等)を安定化するため、MTF を“パッケージ実行”し、
  # かつトップ階層モジュール(dicom_io等)が見えるよう PYTHONPATH を明示追加
  export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

  python3 -m MTF.viz_sample \
    --lr_dir "${LR_DIR}" \
    --sr_dir "${SR_DIR}" \
    --hr_dir "${HR_DIR}" \
    --out_dir "${OUT_DIR}" \
    --series_id "${SERIES_ID}" \
    --sample_index "${SAMPLE_INDEX}" \
    --use_cuda "${USE_CUDA}" \
    --seed "${SEED}" \
    "${SR_SCALE_ARG[@]}" \
    "${LR_SPACING_ARG[@]}" \
    "${HR_SPACING_ARG[@]}"

else
  # ------ 通常モード（本線） ------
  python3 "${SCRIPT_DIR}/main_mtf.py" \
    --lr_dir "${LR_DIR}" \
    --sr_dir "${SR_DIR}" \
    --hr_dir "${HR_DIR}" \
    --out_dir "${OUT_DIR}" \
    --num_rois "${NUM_ROIS}" \
    --draw_nyquist "${DRAW_NYQUIST}" \
    --use_cuda "${USE_CUDA}" \
    "${SR_SCALE_ARG[@]}" \
    "${LR_SPACING_ARG[@]}" \
    "${HR_SPACING_ARG[@]}" \
    "${X_NORM_ARG[@]}" \
    --seed "${SEED}"
fi
