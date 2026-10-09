#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_MANIFEST="${1:?Usage: bash examples/infer_instructttseval.sh INPUT.parquet [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-${PWD}/votspeech_instructttseval_zh}"
GPU_IDS="${GPU_IDS:-0}"

IFS=',' read -r -a votspeech_gpu_ids <<<"${GPU_IDS}"
NUM_GPUS="${NUM_GPUS:-${#votspeech_gpu_ids[@]}}"

PYTHON_BIN="${PYTHON_BIN:-python}" \
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${REPO_ROOT}}" \
INPUT_MANIFEST="${INPUT_MANIFEST}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
RESULTS_MANIFEST="${RESULTS_MANIFEST:-${OUTPUT_DIR}/results.jsonl}" \
GPU_IDS="${GPU_IDS}" \
NUM_GPUS="${NUM_GPUS}" \
TTS_LANGUAGE="${TTS_LANGUAGE:-zh}" \
SEED="${SEED:-1542}" \
bash "${REPO_ROOT}/inference/scripts/run_batch_infer_voice_design.sh"
