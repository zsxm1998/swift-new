#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <results_dir>" >&2
  exit 1
fi

results_dir="$1"

python projects/CGPO/script/visualize_bbox_distribution.py "${results_dir}/z02_thumbnail_choice_think_loc.jsonl" &
python projects/CGPO/script/visualize_bbox_distribution.py "${results_dir}/z07_patch_subtyping_think_loc.jsonl" &
python projects/CGPO/script/visualize_bbox_distribution.py "${results_dir}/z08_patch_grading_think_loc.jsonl" &

wait
