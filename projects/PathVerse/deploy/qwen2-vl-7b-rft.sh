# pipeline_parallel_size设置为2时推理速度显著慢于1
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MAX_PIXELS=$((336*336)) \
swift app \
    --model zsxm_checkpoint/nips/3_rft/qwen2-vl-7b_0306_full_VAL_2_v0#sft#full_bs112_temp1.1_VAL/v0-20250331-211233/checkpoint-1090 \
    --studio_title OmniPT-Qwen2-VL-7B-RFT \
    --stream true \
    --lang zh \
    --infer_backend vllm \
    --gpu_memory_utilization 0.95 \
    --max_model_len 8192 \
    --max_new_tokens 8192 \
    --limit_mm_per_prompt '{"image": 5, "video": 2}' \
    --max_pixels $((336*336)) \
    --pipeline_parallel_size 1 \
    --tensor_parallel_size 4 \
    --temperature 1 \
    --server_port 8200
    