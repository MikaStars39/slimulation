#! /bin/bash

set -exo pipefail
ulimit -n 65535

export HF_ENDPOINT="https://hf-mirror.com"

# Model and Data
BASE_MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/PeRL/outputs/20260110_173129_gspo_qwen30ba3b/iter_0000223_hf"
DATASET="aime2024@32"
CACHE_DIR="./.cache"

# Output
OUTPUT_DIR="./outputs/20260110_173129_gspo_qwen30ba3b_0000223_slime"

# Inference Config
MODE="infer" 
DP_SIZE=8
TP_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
BATCH_SIZE=256
MAX_CONCURRENCY=256
# Sampling Params
MAX_NEW_TOKENS="30000"
TEMPERATURE="0.7"
TOP_P="0.9"

mkdir -p "${OUTPUT_DIR}"

python eval.py \
  --mode "${MODE}" \
  --result-dir "${OUTPUT_DIR}" \
  --model "${BASE_MODEL_PATH}" \
  --dataset "${DATASET}" \
  --cache-dir "${CACHE_DIR}" \
  --dp-size "${DP_SIZE}" \
  --tp-size "${TP_SIZE}" \
  --max-concurrency "${MAX_CONCURRENCY}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --batch-size "${BATCH_SIZE}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"
