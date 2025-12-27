#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <LOG_DIR> [--think]"
  exit 1
fi

# Assign the first argument to LOG_DIR
LOG_DIR="$1"
shift  # Shift the arguments so we can parse the rest

TASK_NAME="02_thumbnail_choice_additional"
RES_FILE="$LOG_DIR/$TASK_NAME.log"
QUESTION_FILE="projects/PathVerse/dataset/nips/9_test/$TASK_NAME.json"
ANSWER_FILE="$LOG_DIR/z$TASK_NAME.jsonl"
VIS_DIR="$LOG_DIR/vis/$TASK_NAME"

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --think)
      THINK_FLAG=true
      shift
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

# Perform overall evaluation
echo -e '—————————————————————————————————— Overall Performance ——————————————————————————————————' > "$RES_FILE"
python projects/PathVerse/eval_codes/nips_eval/choice_eval.py \
  --result_file "$ANSWER_FILE" >> "$RES_FILE"

# Perform per dataset evaluation
DATASETS=("HCC_grading" "ZheYi0607/liver_cancer_thumbnails" "ZheYi0607/ICC_subtype_thumbnails" "RCC")
for DATASET in "${DATASETS[@]}"; do
  echo -e "\n—————————————————————————————————— $DATASET Performance ——————————————————————————————————" >> "$RES_FILE"
  python projects/PathVerse/eval_codes/nips_eval/choice_eval.py \
    --result_file "$ANSWER_FILE" \
    --gt_file "$QUESTION_FILE" \
    --dataset "$DATASET" >> "$RES_FILE"
done
