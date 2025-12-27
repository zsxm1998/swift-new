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

TASK_NAME="05_vessel_nerve_lymph_det_seg"
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
  ANSWER_FILE="${ANSWER_FILE%.jsonl}_think.jsonl"
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

# Perform per dataset no class detection evaluation
echo -n "" > "$RES_FILE"
DATASETS=("vessel_seg" "NI_det" "LN_det")
for DATASET in "${DATASETS[@]}"; do
  echo -e "\n—————————————————————————————————— $DATASET No Class Detection Performance ——————————————————————————————————" >> "$RES_FILE"
  python projects/PathVerse/eval_codes/nips_eval/no_class_detection.py \
    --result_file "$ANSWER_FILE" \
    --gt_file "$QUESTION_FILE" \
    --dataset "$DATASET" \
    --vis_dir "$VIS_DIR/no_class_detection/$DATASET" >> "$RES_FILE"
done

# Perform per dataset with class detection evaluation
DATASETS=("NI_det")
for DATASET in "${DATASETS[@]}"; do
  echo -e "\n—————————————————————————————————— $DATASET With Class Detection Performance ——————————————————————————————————" >> "$RES_FILE"
  python projects/PathVerse/eval_codes/nips_eval/with_class_detection.py \
    --result_file "$ANSWER_FILE" \
    --gt_file "$QUESTION_FILE" \
    --dataset "$DATASET" \
    --vis_dir "$VIS_DIR/with_class_detection/$DATASET" >> "$RES_FILE"
done

# Perform per dataset evaluation
DATASETS=("NI_det")
for DATASET in "${DATASETS[@]}"; do
  echo -e "\n—————————————————————————————————— $DATASET Segmentation Performance ——————————————————————————————————" >> "$RES_FILE"
  python projects/PathVerse/eval_codes/nips_eval/no_class_segmentation.py \
    --result_file "$ANSWER_FILE" \
    --gt_file "$QUESTION_FILE" \
    --dataset "$DATASET" \
    --vis_dir "$VIS_DIR/seg/$DATASET" >> "$RES_FILE"
done
