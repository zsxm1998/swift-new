#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [-ng] [--think] [--func] [-tp <int>] [--batch-size <int>]"
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

TASK_NAME="02_thumbnail_choice"
RES_FILE="$LOG_DIR/$TASK_NAME.log"
QUESTION_FILE="projects/PathVerse/dataset/nips/9_test/$TASK_NAME.json"
ANSWER_FILE="$LOG_DIR/z$TASK_NAME.jsonl"
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
    --func)
      FUNC_FLAG=true
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

# Add think and func for ANSWER_FILE
if [ -n "$THINK_FLAG" ]; then
  ANSWER_FILE="${ANSWER_FILE%.jsonl}_think.jsonl"
fi
if [ -n "$FUNC_FLAG" ]; then
  ANSWER_FILE="${ANSWER_FILE%.jsonl}_func.jsonl"
fi

# Build Python arguments
PYTHON_ARGS=(
  --model-path "$CKPT_DIR"
  --question-file "$QUESTION_FILE"
  --answers-file "$ANSWER_FILE"
)

# Add optional flags
if [ -n "$THINK_FLAG" ]; then
  PYTHON_ARGS+=(--think)
  RES_FILE="${RES_FILE%.log}_think.log"
fi
if [ -n "$FUNC_FLAG" ]; then
  PYTHON_ARGS+=(--func)
  RES_FILE="${RES_FILE%.log}_func.log"
fi
if [ -n "$TP_VALUE" ]; then
  PYTHON_ARGS+=(-tp "$TP_VALUE")
fi
if [ -n "$BATCH_SIZE" ]; then
  PYTHON_ARGS+=(--batch-size "$BATCH_SIZE")
fi

# Run inference if not skipped
if [ -z "$SKIP_PYTHON" ]; then
  python projects/PathVerse/eval_codes/nips_infer.py "${PYTHON_ARGS[@]}"
fi

# Perform overall evaluation
echo -e '—————————————————————————————————— Overall Performance ——————————————————————————————————' > "$RES_FILE"
python projects/PathVerse/eval_codes/nips_eval/choice_eval.py \
  --result_file "$ANSWER_FILE" >> "$RES_FILE"

# Perform per dataset evaluation
DATASETS=("liverWSI" "HCC_grading" "ZheYi0607/liver_cancer_thumbnails" "ZheYi0607/ICC_subtype_thumbnails" "RCC" "LiWeihan/Lung1000" "LiWeihan/Stomach1000")
for DATASET in "${DATASETS[@]}"; do
  echo -e "\n—————————————————————————————————— $DATASET Performance ——————————————————————————————————" >> "$RES_FILE"
  python projects/PathVerse/eval_codes/nips_eval/choice_eval.py \
    --result_file "$ANSWER_FILE" \
    --gt_file "$QUESTION_FILE" \
    --dataset "$DATASET" >> "$RES_FILE"
done
