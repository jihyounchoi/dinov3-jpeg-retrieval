#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_ROOT}/outputs/.cache/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${PROJECT_ROOT}/outputs/.cache/numba}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/outputs/.cache/xdg}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}"

PYTHON="${PYTHON:-python}"
DOWNLOAD_FLAG="${DOWNLOAD_FLAG:-}"

"${PYTHON}" scripts/prepare_data.py --mode full --compress ${DOWNLOAD_FLAG}

"${PYTHON}" scripts/extract_features.py --mode full --model dinov3 --gpu "${GPU}" --device "${DEVICE}"
"${PYTHON}" scripts/extract_features.py --mode full --model dinov3_vits16 --gpu "${GPU}" --device "${DEVICE}"
"${PYTHON}" scripts/extract_features.py --mode full --model resnet50 --gpu "${GPU}" --device "${DEVICE}"
"${PYTHON}" scripts/extract_features.py --mode full --model vit_b_16 --gpu "${GPU}" --device "${DEVICE}"

"${PYTHON}" scripts/evaluate_retrieval.py --mode full
"${PYTHON}" scripts/make_figures.py --mode full
