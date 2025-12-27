#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <LOG_DIR> [--think]"
  exit 1
fi

# Assign the first argument to LOG_DIR
LOG_DIR="$1"
shift  # Shift the arguments so we can parse the rest

TASK_NAME="05_vessel_nerve_lymph_det_seg"
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
