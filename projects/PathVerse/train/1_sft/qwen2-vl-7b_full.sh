# ckpt命名规则：训练类型/模型_数据集_训练方式（full、lora）_模型训练部分（V：vit，A：aligner，L：llm）_epoch数
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
NPROC_PER_NODE=8 \
swift sft \
    --output_dir ./outputs/PathVerse/1_sft/qwen2-vl-7b-base_0625+rsn-0108_full_VAL_2 \
    --model Qwen/Qwen2-VL-7B \
    --deepspeed zero3 \
    --torch_dtype bfloat16 \
    --train_type full \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --freeze_vit false \
    --freeze_aligner false \
    --dataset ./projects/PathVerse/dataset/nips/1_sft/SFT-0625.json \
              ./datasets/RSN_LOC/train_convs_20260108.json \
              swift/self-cognition#500 \
    --padding_free True \
    --max_length 8192 \
    --max_pixels $((1280*28*28)) \
    --truncation_strategy right \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --use_hf false \
    --model_author "浙江大学VIPA实验室" "Zhejiang University VIPA Laboratory" \
    --model_name "PathVerse" "PathVerse"
