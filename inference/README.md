# VoTSpeech inference runtime

This directory contains the source snapshot required to run the VoTSpeech
checkpoint. It is derived from [`dots.tts`](https://github.com/studio-dots-ai/dots.tts)
and adds the instruction-conditioned voice-design branch used by VoTSpeech.

## Installation

Create a clean Python 3.10–3.12 environment, then install the dependencies from
the repository root:

```bash
pip install -r requirements.txt
```

## Batch inference

`MODEL_NAME_OR_PATH` may be a downloaded checkpoint directory or a Hugging
Face repository ID. JSONL input accepts either a top-level `instruction` field
or InstructTTSEval-style `APS`, `DSD`, and `RP` fields.

```bash
MODEL_NAME_OR_PATH=/path/to/VoTSpeech \
INPUT_MANIFEST=examples/voice_design.jsonl \
OUTPUT_DIR=outputs/example \
GPU_IDS=0 \
NUM_GPUS=1 \
TTS_LANGUAGE=zh \
bash inference/scripts/run_batch_infer_voice_design.sh
```

For independent multi-GPU inference, provide comma-separated GPU IDs and set
`NUM_GPUS` to the same count. The runner does not create a DDP/NCCL process
group; each GPU owns one model process. Existing non-empty WAV files are
skipped by default.

```bash
MODEL_NAME_OR_PATH=/path/to/VoTSpeech \
INPUT_MANIFEST=/path/to/zh.parquet \
OUTPUT_DIR=outputs/instructttseval \
GPU_IDS=0,1,2,3 \
NUM_GPUS=4 \
TTS_LANGUAGE=zh \
SEED=1542 \
bash inference/scripts/run_batch_infer_voice_design.sh
```

## Input format

Single instruction:

```json
{"id":"example_zh","text":"今天阳光很好，我们一起出去走走吧。","instruction":"一位年轻女性，声音温暖明亮，语速自然，带着轻松愉快的情绪。"}
```

InstructTTSEval-style rows may provide string-valued `APS`, `DSD`, and `RP`
columns. The Parquet reader projects only `id`, `text`, `APS`, `DSD`, and `RP`;
embedded reference audio is not loaded.

## License and attribution

The runtime is released under Apache-2.0. It is derived from `dots.tts`; retain
the included license and upstream attribution when redistributing modified
versions.
