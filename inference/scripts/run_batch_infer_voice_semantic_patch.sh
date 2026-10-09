#!/usr/bin/env bash
# Guard against accidentally evaluating the older non-semantic checkpoint.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${MODEL_PATH:-}}"
if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  echo "Set MODEL_NAME_OR_PATH to a semantic-patch export." >&2
  exit 1
fi
"${PYTHON_BIN:-python}" -c '
import json, pathlib, sys
config = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
voice = config.get("voice_design") or {}
if not (voice.get("enabled") and voice.get("instruction_conditioning") and voice.get("semantic_patch_conditioning")):
    raise SystemExit("This is not a semantic-patch checkpoint; use run_batch_infer_voice_design.sh for the control/older models")
' "${MODEL_NAME_OR_PATH}"
export MODEL_NAME_OR_PATH
exec bash "${REPO_ROOT}/scripts/run_batch_infer_voice_design.sh" "$@"
