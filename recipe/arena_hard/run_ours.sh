OUTPUT_DIR="/mnt/llm-train/users/explore-train/qingyu/data/arena_hard"
MODEL_NAME="sft-16k-v2"  # Set your model name here
MODEL_PATH="/mnt/llm-train/users/explore-train/qingyu/ckpt/sft-16k-v2"
BASELINE_MODEL="o3-mini-2025-01-31"  # Change to your baseline model name
JUDGE_MODEL="gpt-5"  # or "gemini-2.5"
CATEGORY="hard_prompt"  # or "creative_writing", "coding", "math"
API_BASE="http://ai-api.jdcloud.com/v1"
CONTROL_FEATURES=""  # e.g. "length markdown" to enable style control
echo "OPENAI_API_KEY: $OPENAI_API_KEY"

# Step 1: Preprocess questions with chat template
python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/arena_hard/preprocess.py \
    --input $OUTPUT_DIR/question.jsonl \
    --output $OUTPUT_DIR/preprocess.jsonl \
    --system-prompt "You are JoyAI, a large language model trained by JD (京东). Answer as concisely as possible." \
    --tokenizer $MODEL_PATH

# Step 2: Run batch inference
python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/arena_hard/inference.py \
    --input "$OUTPUT_DIR/preprocess.jsonl" \
    --output "$OUTPUT_DIR/results.jsonl" \
    --model "$MODEL_PATH" \
    --tp-size 1 \
    --dp-size 8 \
    --temperature 0.6 \
    --top-p 1 \
    --max-tokens 32768 \
    --resume

# Step 3: Post-process results with metadata
python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/arena_hard/postprocess.py \
    --input "$OUTPUT_DIR/results.jsonl" \
    --output "$OUTPUT_DIR/${MODEL_NAME}.jsonl" \
    --model-name "$MODEL_NAME" \
    --tokenizer "$MODEL_PATH"

# Step 4: API Judge - Compare with baseline model
python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/arena_hard/api_judge.py \
    --questions "$OUTPUT_DIR/question.jsonl" \
    --model-a "$OUTPUT_DIR/${BASELINE_MODEL}.jsonl" \
    --model-b "$OUTPUT_DIR/${MODEL_NAME}.jsonl" \
    --output "$OUTPUT_DIR/judgments/${JUDGE_MODEL}/${MODEL_NAME}.jsonl" \
    --judge-model "$JUDGE_MODEL" \
    --api-base "$API_BASE" \
    --api-key "$OPENAI_API_KEY" \
    --temperature 1 \
    --max-tokens 2048 \
    --parallel 64 \
    --timeout 3600 \
    --category "$CATEGORY" \
    --resume

# Step 5: Calculate scores and show leaderboard
if [ -n "$CONTROL_FEATURES" ]; then
    python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/arena_hard/show_result.py \
        --judgment-file "$OUTPUT_DIR/judgments/${JUDGE_MODEL}/${MODEL_NAME}.jsonl" \
        --baseline-model "$BASELINE_MODEL" \
        --answer-dir "$OUTPUT_DIR" \
        --control-features $CONTROL_FEATURES \
        --output "$OUTPUT_DIR/leaderboard_${JUDGE_MODEL}.json"
else
    python /mnt/llm-train/users/explore-train/qingyu/slimulation/recipe/arena_hard/show_result.py \
        --judgment-file "$OUTPUT_DIR/judgments/${JUDGE_MODEL}/${MODEL_NAME}.jsonl" \
        --baseline-model "$BASELINE_MODEL" \
        --output "$OUTPUT_DIR/leaderboard_${JUDGE_MODEL}.json"
fi