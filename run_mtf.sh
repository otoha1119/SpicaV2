#!/usr/bin/env bash

# Wrapper script for running the MTF extraction pipeline.
#
# This script parses command‑line flags and forwards them to the
# Python entry point.  GPU visibility can be controlled with
# --use_cuda and --gpu_ids.  When --use_cuda=0 or when gpu_ids is -1
# the code will fall back to CPU execution.  Series directories and
# output location must be provided.

set -euo pipefail

usage() {
    echo "Usage: $0 --lr_dir DIR --sr_dir DIR --hr_dir DIR [options]"
    echo
    echo "Required arguments:"
    echo "  --lr_dir DIR         Path to directory containing LR DICOM files"
    echo "  --sr_dir DIR         Path to directory containing SR DICOM files"
    echo "  --hr_dir DIR         Path to directory containing HR DICOM files"
    echo
    echo "Optional arguments:"
    echo "  --out_dir DIR        Directory to write outputs (default: outputs)"
    echo "  --num_rois N         Number of ROIs per series (default: 150)"
    echo "  --draw_nyquist [0|1] Draw Nyquist lines in plot (default: 1)"
    echo "  --use_cuda [0|1]     Enable CUDA FFT when available (default: 1)"
    echo "  --gpu_ids IDS        Comma‑separated CUDA device IDs (default: 0)"
    echo "  --sr_scale S         Scale factor to correct SR PixelSpacing (optional)"
    echo "  --seed S             RNG seed (default: 42)"
    echo "  -h, --help           Show this help message"
}

# Default values
OUT_DIR="outputs"
NUM_ROIS=150
DRAW_NYQUIST=1
USE_CUDA=1
GPU_IDS="0"
SR_SCALE=""
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --lr_dir)
            LR_DIR="$2"; shift; shift ;;
        --sr_dir)
            SR_DIR="$2"; shift; shift ;;
        --hr_dir)
            HR_DIR="$2"; shift; shift ;;
        --out_dir)
            OUT_DIR="$2"; shift; shift ;;
        --num_rois)
            NUM_ROIS="$2"; shift; shift ;;
        --draw_nyquist)
            DRAW_NYQUIST="$2"; shift; shift ;;
        --use_cuda)
            USE_CUDA="$2"; shift; shift ;;
        --gpu_ids)
            GPU_IDS="$2"; shift; shift ;;
        --sr_scale)
            SR_SCALE="$2"; shift; shift ;;
        --seed)
            SEED="$2"; shift; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $key"; usage; exit 1 ;;
    esac
done

# Check required args
if [[ -z "${LR_DIR:-}" || -z "${SR_DIR:-}" || -z "${HR_DIR:-}" ]]; then
    echo "Error: --lr_dir, --sr_dir and --hr_dir are required" >&2
    usage
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

python3 "$(dirname "$0")/main_mtf.py" \
    --lr_dir "${LR_DIR}" \
    --sr_dir "${SR_DIR}" \
    --hr_dir "${HR_DIR}" \
    --out_dir "${OUT_DIR}" \
    --num_rois "${NUM_ROIS}" \
    --draw_nyquist "${DRAW_NYQUIST}" \
    --use_cuda "${USE_CUDA}" \
    ${SR_SCALE:+--sr_scale "$SR_SCALE"} \
    --seed "${SEED}"