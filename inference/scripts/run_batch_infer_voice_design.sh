#!/usr/bin/env bash
# Multi-GPU voice-design batch inference without DDP/NCCL.
#
#   MODEL_NAME_OR_PATH=/path/to/exports/step-00010000 \
#   INPUT_MANIFEST=/path/to/example.jsonl-or-zh.parquet \
#   OUTPUT_DIR=samples/step-00010000 \
#   NUM_GPUS=8 \
#   bash scripts/run_batch_infer_voice_design.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYBIN="${PYTHON_BIN:-python}"

if [[ ! -x "${PYBIN}" ]]; then
  requested_python="${PYBIN}"
  if ! PYBIN="$(command -v "${requested_python}")"; then
    echo "PYTHON_BIN is not executable or available on PATH: ${requested_python}" >&2
    exit 1
  fi
fi

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${MODEL_PATH:-}}"
INPUT_MANIFEST="${INPUT_MANIFEST:-}"
OUTPUT_DIR="${OUTPUT_DIR:-samples/voice_design_batch}"
NUM_GPUS="${NUM_GPUS:-8}"

if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  echo "Set MODEL_NAME_OR_PATH to an exported inference checkpoint." >&2
  exit 1
fi
if [[ -z "${INPUT_MANIFEST}" ]]; then
  echo "Set INPUT_MANIFEST to an input JSONL or Parquet file." >&2
  exit 1
fi
# Absolute and explicitly relative values are local paths and can be checked
# eagerly. Other owner/repository values are accepted as Hugging Face repo IDs
# and resolved by the Python runtime.
if [[ -e "${MODEL_NAME_OR_PATH}" ]]; then
  :
elif [[ "${MODEL_NAME_OR_PATH}" == /* || "${MODEL_NAME_OR_PATH}" == ./* || "${MODEL_NAME_OR_PATH}" == ../* ]]; then
  echo "Model path not found: ${MODEL_NAME_OR_PATH}" >&2
  exit 1
elif [[ "${MODEL_NAME_OR_PATH}" != */* ]]; then
  echo "MODEL_NAME_OR_PATH must be an existing local path or a Hugging Face owner/repository ID." >&2
  exit 1
fi
if [[ ! -f "${INPUT_MANIFEST}" ]]; then
  echo "Input manifest not found: ${INPUT_MANIFEST}" >&2
  exit 1
fi
if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: ${NUM_GPUS}" >&2
  exit 1
fi

ARGS=(
  "${PYBIN}" -m dots_tts.batch_cli
  --model-name-or-path "${MODEL_NAME_OR_PATH}"
  --input-manifest "${INPUT_MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --num-gpus "${NUM_GPUS}"
  --variants "${VARIANTS:-APS,DSD,RP}"
  --template-name "${TEMPLATE_NAME:-voice_design}"
  --precision "${PRECISION:-bfloat16}"
  --speaker-scale "${SPEAKER_SCALE:-1.5}"
  --max-generate-length "${MAX_GENERATE_LENGTH:-500}"
  --max-sequence-length "${MAX_SEQUENCE_LENGTH:-2048}"
  --model-load-timeout-seconds "${MODEL_LOAD_TIMEOUT_SECONDS:-900}"
  --seed "${SEED:-42}"
  --worker-log-level "${WORKER_LOG_LEVEL:-WARNING}"
  --output-subtype "${OUTPUT_SUBTYPE:-PCM_16}"
)

# Unset uses the exported checkpoint's voice sampler configuration. To
# reproduce older batches, explicitly set 32 steps and guidance 2.0.
if [[ -n "${VOICE_NUM_STEPS:-}" ]]; then ARGS+=(--voice-num-steps "${VOICE_NUM_STEPS}"); fi
if [[ -n "${VOICE_GUIDANCE_SCALE:-}" ]]; then ARGS+=(--voice-guidance-scale "${VOICE_GUIDANCE_SCALE}"); fi
if [[ -n "${GPU_IDS:-}" ]]; then ARGS+=(--gpu-ids "${GPU_IDS}"); fi
if [[ -n "${RESULTS_MANIFEST:-}" ]]; then
  ARGS+=(--results-manifest "${RESULTS_MANIFEST}")
fi
if [[ -n "${ODE_METHOD:-}" ]]; then ARGS+=(--ode-method "${ODE_METHOD}"); fi
if [[ -n "${NUM_STEPS:-}" ]]; then ARGS+=(--num-steps "${NUM_STEPS}"); fi
if [[ -n "${GUIDANCE_SCALE:-}" ]]; then
  ARGS+=(--guidance-scale "${GUIDANCE_SCALE}")
fi
# LANGUAGE is a standard locale variable (often zh_CN:zh:en_US:en), not a TTS
# language selector. Use a task-specific name so locale settings cannot leak
# into inference requests.
if [[ -n "${TTS_LANGUAGE:-}" ]]; then ARGS+=(--language "${TTS_LANGUAGE}"); fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then ARGS+=(--max-samples "${MAX_SAMPLES}"); fi
if [[ "${NORMALIZE_TEXT:-0}" == "1" ]]; then ARGS+=(--normalize-text); fi
if [[ "${RESUME:-1}" == "0" ]]; then ARGS+=(--no-resume); fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then ARGS+=(--dry-run); fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "Launching voice-design batch inference"
echo "  model:      ${MODEL_NAME_OR_PATH}"
echo "  input:      ${INPUT_MANIFEST}"
echo "  output:     ${OUTPUT_DIR}"
echo "  GPU count:  ${NUM_GPUS}"
exec "${ARGS[@]}"
