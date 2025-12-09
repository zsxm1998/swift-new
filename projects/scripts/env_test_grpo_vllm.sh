# 当设置zero3时，要设置 --vllm_enforce_eager true 或者设置环境变量export TORCHDYNAMO_DISABLE=1，否则会报错
# 这是因为vLLM和zero3冲突，设置zero2则不用设置这个参数
MAX_PIXELS=1003520 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --output_dir ./outputs/env_test \
    --train_type full \
    --deepspeed zero3 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --dataset 'hf::hiyouga/geometry3k:train' \
    --val_dataset 'hf::hiyouga/geometry3k:validation' \
    --load_from_cache_file true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_tensor_parallel_size 4 \
    --vllm_enforce_eager true \
    --system examples/train/grpo/prompt.txt \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_completion_length 4096 \
    --max_length 8192 \
    --reward_funcs accuracy format \
    --num_generations 8 \
    --sleep_level 0 \
    --temperature 1.0 \
    --top_p 0.85 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --log_completions true
