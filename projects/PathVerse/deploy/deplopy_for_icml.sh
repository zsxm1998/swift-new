CUDA_VISIBLE_DEVICES=7 \
MAX_PIXELS=$((1280*28*28)) \
IMAGE_MAX_TOKEN_NUM=1280 \
swift app \
    --model lingshu-medical-mllm/Lingshu-7B \
    --model_type qwen2_5_vl \
    --studio_title "ICML2026" \
    --stream true \
    --lang zh \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --max_new_tokens 8192 \
    --vllm_limit_mm_per_prompt '{"image": 10, "video": 2}' \
    --vllm_engine_kwargs '{"compilation_config": {"mode": "none"}}' \
    --max_pixels $((1280*28*28)) \
    --vllm_pipeline_parallel_size 1 \
    --vllm_tensor_parallel_size 1 \
    --temperature 1 \
    --server_port 8100


# Qwen/Qwen2.5-VL-7B-Instruct
# Qwen/Qwen3-VL-8B-Instruct
# OpenGVLab/InternVL3_5-8B-Instruct
# lingshu-medical-mllm/Lingshu-7B