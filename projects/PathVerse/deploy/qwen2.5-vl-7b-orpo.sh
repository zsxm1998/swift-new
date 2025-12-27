# pipeline_parallel_size设置为2时推理速度显著慢于1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=$((1280*28*28)) \
swift app \
    --model zsxm_checkpoint/nips/2_dpo/ORPO_qwen2.5-vl-7b_0415_full_VAL_2_v0#sft#full_VAL_1/v0-20250430-142917/checkpoint-21 \
    --studio_title OmniPT-Qwen2.5-VL-7B-ORPO \
    --stream true \
    --lang zh \
    --infer_backend vllm \
    --gpu_memory_utilization 0.95 \
    --max_model_len 8192 \
    --max_new_tokens 8192 \
    --limit_mm_per_prompt '{"image": 5, "video": 2}' \
    --max_pixels $((1280*28*28)) \
    --pipeline_parallel_size 1 \
    --tensor_parallel_size 4 \
    --temperature 1 \
    --server_port 8201
    