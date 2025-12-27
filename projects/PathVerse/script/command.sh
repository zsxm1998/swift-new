bash projects/PathVerse/script/eval/01_thumbnail_seg.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208; bash projects/PathVerse/script/eval/07_patch_subtyping.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208

bash projects/PathVerse/script/eval/02_thumbnail_choice.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208; bash projects/PathVerse/script/eval/02_thumbnail_choice.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208 --think; bash projects/PathVerse/script/eval/83_ShanghaiBreast_patch_choice.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208

bash projects/PathVerse/script/eval/03_nucleus_det_no_class.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208; bash projects/PathVerse/script/eval/02_thumbnail_choice.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208 --think --func

bash projects/PathVerse/script/eval/04_nucleus_det_with_class.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208; bash projects/PathVerse/script/eval/02_thumbnail_choice.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208 --func

bash projects/PathVerse/script/eval/05_vessel_nerve_lymph_det_seg.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208; bash projects/PathVerse/script/eval/08_patch_grading.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208

bash projects/PathVerse/script/eval/06_cancer_in_vessel_nerve_lymph.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208; bash projects/PathVerse/script/eval/09_classify_nucleus_bbox.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208; bash projects/PathVerse/script/eval/09_classify_nucleus_bbox.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208 --think

bash projects/PathVerse/script/eval/81_ShanghaiBreast_thumbnail_seg.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208; bash projects/PathVerse/script/eval/82_ShanghaiBreast_thumbnail_choice.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208; bash projects/PathVerse/script/eval/07_patch_subtyping.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208 --think

bash projects/PathVerse/script/eval/08_patch_grading.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208 --think








# --------------------------- 测试使用CVPR的attention提取的patch对缩略图的效果 ---------------------------
bash projects/PathVerse/script/eval/93_thumbnail_choice_tool.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208 --func; bash projects/PathVerse/script/eval/91_thumbnail_choice_additional_tool.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208 --func

bash projects/PathVerse/script/eval/94_thumbnail_choice_notool.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208; bash projects/PathVerse/script/eval/92_thumbnail_choice_additional_notool.sh outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208


# --------------------------- 推理训练集 ---------------------------
python projects/PathVerse/eval_codes/nips_infer.py --question-file "zsxm_dataset/nips/3_rft/08_thumbnail_func_point.json" --answers-file "zsxm_val/results/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048|v0|1137/train_08_thumbnail_func_point_think_10.jsonl" --model-path "outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208" --ignore-origin-system --func --think --seed 10; python projects/PathVerse/eval_codes/nips_infer.py --question-file "zsxm_dataset/nips/3_rft/08_thumbnail_func_point.json" --answers-file "zsxm_val/results/nips/3_rft/qwen2-vl-7b-base_0625#sft#FindPointS2ThNh_G16_bs128_max-token-2048|v0|1137/train_08_thumbnail_func_point_10.jsonl" --model-path "outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208" --ignore-origin-system --func --seed 10

python projects/PathVerse/eval_codes/nips_infer.py --question-file "zsxm_dataset/nips/3_rft/02_thumbnail_choice.json" --answers-file "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_02_thumbnail_choice_07.jsonl" --model-path "outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208" --seed 7; python projects/PathVerse/eval_codes/nips_infer.py --question-file "zsxm_dataset/nips/3_rft/07_private_patch_choice.json" --answers-file "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_07_private_patch_choice_07.jsonl" --model-path "outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208" --seed 7


# --------------------- 计算delta ---------------------
python projects/PathVerse/eval_codes/calculate_delta.py --model outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208 --pivot zsxm_dataset/nips/3_rft/02_thumbnail_choice.json --jsonl "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_02_thumbnail_choice_09.jsonl" "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_02_thumbnail_choice_10.jsonl" --output "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_02_thumbnail_choice_delta_4.jsonl"

python projects/PathVerse/eval_codes/calculate_delta.py --model outputs/PathVerse/1_sft/qwen3-vl-8b-thinking_0625_full_VAL_2/v0-20251018-163841/checkpoint-11208 --pivot zsxm_dataset/nips/3_rft/07_private_patch_choice.json --jsonl "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_07_private_patch_choice_08.jsonl" "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_07_private_patch_choice_09.jsonl" "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_07_private_patch_choice_10.jsonl" --output "zsxm_val/results/nips/1_sft/qwen2-vl-7b-base_0625_full_VAL_2|v0|11206/train_07_private_patch_choice_delta_4.jsonl"
