CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output_dir ./outputs/env_test \
    --train_type full \
    --deepspeed zero2 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --dataset 'hf::PAPOGalaxy/PAPO_ViRL39K_train' \
    --val_dataset 'hf::PAPOGalaxy/PAPO_MMK12_test' \
    --dataset_num_proc 32 \
    --load_from_cache_file true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --use_hf false
