#!/usr/bin/env bash
# MTF pipeline launcher

set -euo pipefail

### ====================== EDIT HERE (デフォルト設定) ====================== ###
# DICOM ディレクトリ
LR_DIR="/workspace/DataSet/ImageCAS/003.ImgCast"
SR_DIR="/workspace/results/001.ImgCast"
HR_DIR="/workspace/DataSet/photonCT/PhotonCT1024v2/DICOMSAVE-20240514142921-000"

# 出力先
OUT_DIR="/workspace/results"

# ROI/例画像出力
EXPORT_EXAMPLES=1                 # 0=無効, 1=有効
EXAMPLES_PER_SERIES=1             # 1シリーズあたりの例数

# 処理パラメータ
NUM_ROIS=4000                     # 1シリーズあたり抽出するROI数
DRAW_NYQUIST=0                    # 1=Nyquist縦線を描画, 0=非表示
USE_CUDA=1                        # 1=CUDA使用(可能なら), 0=CPU強制
GPU_IDS="0"                       # 例: "0", "0,1", "-1"(CPU扱い)

# SRのPixelSpacing補正倍率（未更新DICOM対策）。空なら補正しない。
SR_SCALE="2.0"                    # 例: "2.0" / ""（空文字で無効）

# x軸正規化 (1=正規化, 0=物理単位[cycles/mm])
X_NORM=1

# PixelSpacing明示上書き（空なら無効）
LR_SPACING_ROW="0.3184"
LR_SPACING_COL="0.3184"
HR_SPACING_ROW="0.136719"
HR_SPACING_COL="0.136719"

# ROI 幾何 & 角度ゲート（★新規）
ROI_WIDTH=40
ROI_HEIGHT=100
ANGLE_MIN=6.0                     # 水平/垂直からの最小乖離(度)
ANGLE_MAX=12.0                    # 最大乖離(度)

# 「1本＆上下貫通」チェック（★新規）
REQUIRE_SINGLE_EDGE=1             # 1=必須/0=無効
EDGE_VERTICAL_TOL_DEG=2.5         # 回転後ROI内での“ほぼ縦”許容(度)
MIN_VERTICAL_SPAN_RATIO=0.92      # ROI高さに対する縦スパン下限
CENTER_TOLERANCE_RATIO=0.08       # ROI中心±割合で通過

# ESF 安定化（★新規）
ESF_FRAC=0.6                      # ROI中央帯の使用割合(0<frac<=1)
ESF_REDUCE="median"               # mean/median/trimmed
TRIM_ALPHA=0.1                    # トリムド平均の両端トリム率(0<=a<0.5)

# 再現性
SEED=42
### ====================================================================== ###

usage() {
  echo "Usage: $0 [--lr_dir DIR --sr_dir DIR --hr_dir DIR] [options]"
  echo
  echo "Common options:"
  echo "  --lr_dir DIR | --sr_dir DIR | --hr_dir DIR | --out_dir DIR"
  echo "  --num_rois N --draw_nyquist 0|1 --use_cuda 0|1 --gpu_ids IDS"
  echo "  --sr_scale S --x_norm 0|1 --lr_spacing ROW COL --hr_spacing ROW COL --seed N"
  echo
  echo "ROI/ESF options (new):"
  echo "  --roi_width W --roi_height H"
  echo "  --angle_min A --angle_max A"
  echo "  --require_single_edge 0|1 --edge_vertical_tol_deg D"
  echo "  --min_vertical_span_ratio R --center_tolerance_ratio R"
  echo "  --esf_frac R --esf_reduce mean|median|trimmed --trim_alpha A"
}

# ------------- オプション解析 -------------
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
    --x_norm)        X_NORM="$2"; shift; shift ;;
    --lr_spacing)    LR_SPACING_ROW="$2"; LR_SPACING_COL="$3"; shift; shift; shift ;;
    --hr_spacing)    HR_SPACING_ROW="$2"; HR_SPACING_COL="$3"; shift; shift; shift ;;
    --seed)          SEED="$2"; shift; shift ;;
    # 新規:
    --roi_width)     ROI_WIDTH="$2"; shift; shift ;;
    --roi_height)    ROI_HEIGHT="$2"; shift; shift ;;
    --angle_min)     ANGLE_MIN="$2"; shift; shift ;;
    --angle_max)     ANGLE_MAX="$2"; shift; shift ;;
    --require_single_edge) REQUIRE_SINGLE_EDGE="$2"; shift; shift ;;
    --edge_vertical_tol_deg) EDGE_VERTICAL_TOL_DEG="$2"; shift; shift ;;
    --min_vertical_span_ratio) MIN_VERTICAL_SPAN_RATIO="$2"; shift; shift ;;
    --center_tolerance_ratio)  CENTER_TOLERANCE_RATIO="$2"; shift; shift ;;
    --esf_frac)      ESF_FRAC="$2"; shift; shift ;;
    --esf_reduce)    ESF_REDUCE="$2"; shift; shift ;;
    --trim_alpha)    TRIM_ALPHA="$2"; shift; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown option: $key"; usage; exit 1 ;;
  esac
done

# ------------- 前提チェック -------------
if [[ -z "${LR_DIR}" || -z "${SR_DIR}" || -z "${HR_DIR}" ]]; then
  echo "Error: LR_DIR / SR_DIR / HR_DIR が未設定です。" >&2
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
  export CUDA_VISIBLE_DEVICES=""
fi

# ------------- 引数組み立て -------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SR_SCALE_ARG=()
[[ -n "${SR_SCALE}" ]] && SR_SCALE_ARG=(--sr_scale "${SR_SCALE}")

LR_SPACING_ARG=()
[[ -n "${LR_SPACING_ROW}" && -n "${LR_SPACING_COL}" ]] && LR_SPACING_ARG=(--lr_spacing "${LR_SPACING_ROW}" "${LR_SPACING_COL}")

HR_SPACING_ARG=()
[[ -n "${HR_SPACING_ROW}" && -n "${HR_SPACING_COL}" ]] && HR_SPACING_ARG=(--hr_spacing "${HR_SPACING_ROW}" "${HR_SPACING_COL}")

# 新規オプション
ROI_ARGS=(
  --roi_width "${ROI_WIDTH}"
  --roi_height "${ROI_HEIGHT}"
  --angle_min "${ANGLE_MIN}"
  --angle_max "${ANGLE_MAX}"
  --require_single_edge "${REQUIRE_SINGLE_EDGE}"
  --edge_vertical_tol_deg "${EDGE_VERTICAL_TOL_DEG}"
  --min_vertical_span_ratio "${MIN_VERTICAL_SPAN_RATIO}"
  --center_tolerance_ratio "${CENTER_TOLERANCE_RATIO}"
  --esf_frac "${ESF_FRAC}"
  --esf_reduce "${ESF_REDUCE}"
  --trim_alpha "${TRIM_ALPHA}"
)

echo "[INFO] LR_DIR=${LR_DIR}"
echo "[INFO] SR_DIR=${SR_DIR}"
echo "[INFO] HR_DIR=${HR_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] NUM_ROIS=${NUM_ROIS}, DRAW_NYQUIST=${DRAW_NYQUIST}"
echo "[INFO] USE_CUDA=${USE_CUDA}, GPU_IDS=${GPU_IDS}"
echo "[INFO] SR_SCALE=${SR_SCALE:-<none>}, X_NORM=${X_NORM}, SEED=${SEED}"
echo "[INFO] LR_SPACING=${LR_SPACING_ROW:-<none>} ${LR_SPACING_COL:-<none>}, HR_SPACING=${HR_SPACING_ROW:-<none>} ${HR_SPACING_COL:-<none>}"
echo "[INFO] ROI: ${ROI_ARGS[*]}"
echo

python3 "${SCRIPT_DIR}/main_mtf.py" \
  --lr_dir "${LR_DIR}" \
  --sr_dir "${SR_DIR}" \
  --hr_dir "${HR_DIR}" \
  --out_dir "${OUT_DIR}" \
  --num_rois "${NUM_ROIS}" \
  --draw_nyquist "${DRAW_NYQUIST}" \
  --use_cuda "${USE_CUDA}" \
  --export_examples "${EXPORT_EXAMPLES}" \
  --examples_per_series "${EXAMPLES_PER_SERIES}" \
  "${SR_SCALE_ARG[@]}" \
  "${LR_SPACING_ARG[@]}" \
  "${HR_SPACING_ARG[@]}" \
  --x_norm "${X_NORM}" \
  --seed "${SEED}" \
  "${ROI_ARGS[@]}"
