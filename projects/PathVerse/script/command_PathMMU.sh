CUDA_VISIBLE_DEVICES=0 bash projects/PathVerse/script/eval/11_PathMMU_val_choice.sh Qwen/Qwen2.5-VL-7B-Instruct

CUDA_VISIBLE_DEVICES=1 bash projects/PathVerse/script/eval/12_PathMMU_test_choice.sh Qwen/Qwen2.5-VL-7B-Instruct

CUDA_VISIBLE_DEVICES=2 bash projects/PathVerse/script/eval/13_PathMMU_test_tiny_choice.sh Qwen/Qwen2.5-VL-7B-Instruct


bash projects/PathVerse/script/eval/11_PathMMU_val_choice.sh llava-hf/llava-onevision-qwen2-7b-ov-hf

bash projects/PathVerse/script/eval/12_PathMMU_test_choice.sh llava-hf/llava-onevision-qwen2-7b-ov-hf

bash projects/PathVerse/script/eval/13_PathMMU_test_tiny_choice.sh llava-hf/llava-onevision-qwen2-7b-ov-hf


CUDA_VISIBLE_DEVICES=6 bash projects/PathVerse/script/eval/11_PathMMU_val_choice.sh lingshu-medical-mllm/Lingshu-7B; CUDA_VISIBLE_DEVICES=6 bash projects/PathVerse/script/eval/12_PathMMU_test_choice.sh lingshu-medical-mllm/Lingshu-7B; CUDA_VISIBLE_DEVICES=6 bash projects/PathVerse/script/eval/13_PathMMU_test_tiny_choice.sh lingshu-medical-mllm/Lingshu-7B

USE_HF=1 CUDA_VISIBLE_DEVICES=3 bash projects/PathVerse/script/eval/11_PathMMU_val_choice.sh ddvd233/QoQ-Med-VL-7B; USE_HF=1 CUDA_VISIBLE_DEVICES=3 bash projects/PathVerse/script/eval/12_PathMMU_test_choice.sh ddvd233/QoQ-Med-VL-7B; USE_HF=1 CUDA_VISIBLE_DEVICES=3 bash projects/PathVerse/script/eval/13_PathMMU_test_tiny_choice.sh ddvd233/QoQ-Med-VL-7B

bash projects/PathVerse/script/eval/11_PathMMU_val_choice.sh outputs/PathVerse/1_sft/qwen2-vl-7b-base_0625+rsn-0108+RD_full_VAL_2/v1-20260118-163824/checkpoint-11804; bash projects/PathVerse/script/eval/12_PathMMU_test_choice.sh outputs/PathVerse/1_sft/qwen2-vl-7b-base_0625+rsn-0108+RD_full_VAL_2/v1-20260118-163824/checkpoint-11804; bash projects/PathVerse/script/eval/13_PathMMU_test_tiny_choice.sh outputs/PathVerse/1_sft/qwen2-vl-7b-base_0625+rsn-0108+RD_full_VAL_2/v1-20260118-163824/checkpoint-11804