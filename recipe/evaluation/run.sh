#!/bin/bash
MODEL_PATH=/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507
CACHE_DIR="/mnt/llm-train/users/explore-train/qingyu/.cache"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/eval_outputs/${TIMESTAMP}_qwen_4b"

# Step 1: Prepare data (load benchmarks and apply chat template)
python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/evaluation/prepare_data.py \
    --dataset "mmlu_pro@1" \
    --cache-dir "$CACHE_DIR" \
    --out-dir "$OUTPUT_DIR" \
    --model "$MODEL_PATH" \
    --prompt-format "lighteval"

# Step 2: Run batch inference
python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/evaluation/inference.py \
    --input "$OUTPUT_DIR/data.chat.jsonl" \
    --output "$OUTPUT_DIR/results.jsonl" \
    --model "$MODEL_PATH" \
    --tp-size 1 \
    --dp-size 8 \
    --temperature 0.6 \
    --top-p 1 \
    --max-tokens 32768 \
    --resume

# Step 3: Evaluate and calculate metrics
python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/evaluation/evaluate.py \
    --input "$OUTPUT_DIR/results.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --num-proc 32
