# $((1280*28*28))
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MAX_PIXELS=$((336*336)) \
swift deploy \
    --model zsxm_checkpoint/nips/3_rft/qwen2-vl-7b_0306_full_VAL_2_v0#sft#full_bs112_temp1.1/v0-20250326-191610/checkpoint-1090 \
    --infer_backend vllm \
    --pipeline_parallel_size 1 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --max_new_tokens 4096 \
    --limit_mm_per_prompt '{"image": 10}' \
    --served_model_name qwen2-vl-7b-rft \
    --port 8000