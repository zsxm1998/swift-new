#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [-ng] [--think] [-tp <int>] [--batch-size <int>]"
  exit 1
fi

# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
shift  # Shift the arguments so we can parse the rest

# Check if the directory exists locally
if [ -d "$CKPT_DIR" ]; then
  if [[ "$(basename "$CKPT_DIR")" != *checkpoint* ]]; then
    echo "Error: CKPT_DIR basename must contain the word 'checkpoint'."
    exit 1
  fi
  CKPT_NAME=$(echo "$CKPT_DIR" | awk -F'/' '{split($(NF-1), v, "-"); split($NF, s, "-"); print $(NF-3)"/"$(NF-2)"|"v[1]"|"s[length(s)]}')
else
  CKPT_NAME="0_baseline/$(basename "$CKPT_DIR")"
fi

# Initialize variables
LOG_DIR="outputs/PathVerse/results/$CKPT_NAME"
mkdir -p "$LOG_DIR"

TASK_NAME="10_refpath_det"
RES_FILE="$LOG_DIR/$TASK_NAME.log"
QUESTION_FILE_A="datasets_public/RefPath/refpath_testA_convs.json"
QUESTION_FILE_B="datasets_public/RefPath/refpath_testB_convs.json"
ANSWER_FILE_A="$LOG_DIR/zrefpath_testA_convs.jsonl"
ANSWER_FILE_B="$LOG_DIR/zrefpath_testB_convs.jsonl"
VIS_DIR="$LOG_DIR/vis/$TASK_NAME"

# Default values (unset initially)
TP_VALUE=""
BATCH_SIZE=""

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -ng)
      SKIP_PYTHON=true
      shift
      ;;
    --think)
      THINK_FLAG=true
      shift
      ;;
    -tp)
      TP_VALUE="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Add think for ANSWER_FILE
if [ -n "$THINK_FLAG" ]; then
  ANSWER_FILE_A="${ANSWER_FILE_A%.jsonl}_think.jsonl"
  ANSWER_FILE_B="${ANSWER_FILE_B%.jsonl}_think.jsonl"
  RES_FILE="${RES_FILE%.log}_think.log"
fi

# Build Python arguments
PYTHON_ARGS=(
  --model-path "$CKPT_DIR"
)

# Add optional flags
if [ -n "$THINK_FLAG" ]; then
  PYTHON_ARGS+=(--think)
fi
if [ -n "$TP_VALUE" ]; then
  PYTHON_ARGS+=(-tp "$TP_VALUE")
fi
if [ -n "$BATCH_SIZE" ]; then
  PYTHON_ARGS+=(--batch-size "$BATCH_SIZE")
fi

# Run inference if not skipped
if [ -z "$SKIP_PYTHON" ]; then
  python projects/PathVerse/eval_codes/nips_infer.py \
    --question-file "$QUESTION_FILE_A" \
    --answers-file "$ANSWER_FILE_A" \
    "${PYTHON_ARGS[@]}"

  python projects/PathVerse/eval_codes/nips_infer.py \
    --question-file "$QUESTION_FILE_B" \
    --answers-file "$ANSWER_FILE_B" \
    "${PYTHON_ARGS[@]}"
fi

# Perform evaluation
echo -e "—————————————————————————————————— RefPath TestA Detection Performance ——————————————————————————————————" > "$RES_FILE"
python projects/PathVerse/eval_codes/nips_eval/refpath_detection.py \
  --result_file "$ANSWER_FILE_A" \
  --gt_file "$QUESTION_FILE_A" \
  --vis_dir "$VIS_DIR/testA" >> "$RES_FILE"

echo -e "\n—————————————————————————————————— RefPath TestB Detection Performance ——————————————————————————————————" >> "$RES_FILE"
python projects/PathVerse/eval_codes/nips_eval/refpath_detection.py \
  --result_file "$ANSWER_FILE_B" \
  --gt_file "$QUESTION_FILE_B" \
  --vis_dir "$VIS_DIR/testB" >> "$RES_FILE"
