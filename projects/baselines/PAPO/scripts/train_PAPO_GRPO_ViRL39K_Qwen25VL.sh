#!/bin/bash
set -x

# 获取模型大小参数，默认为7B
MODEL_SIZE="${1:-7}"

# 清理参数：去除可能的b/B后缀，只保留数字
MODEL_SIZE_NUM=$(echo "$MODEL_SIZE" | tr '[:lower:]' '[:upper:]' | sed 's/[^0-9]//g')

# 检查是否为有效数字
if ! [[ "$MODEL_SIZE_NUM" =~ ^[0-9]+$ ]]; then
    echo "错误：参数必须是数字（如7、3等）或带b/B后缀的数字（如7b、3B等）"
    exit 1
fi

# 从环境变量提取Bark相关参数
EXTRA_ARGS=()
if [[ -n "$BARK_URL" ]]; then
    EXTRA_ARGS+=(--swanlab_bark_url "$BARK_URL")
fi
if [[ -n "$BARK_DEVICE_TOKEN" ]]; then
    EXTRA_ARGS+=(--swanlab_bark_key "$BARK_DEVICE_TOKEN")
fi

#1003520 = 1280*28*28; 200704 = 256*28*28
WANDB_PROJECT="PAPO-Reproduce" \
MAX_PIXELS=1003520 \
MIN_PIXELS=200704 \
NPROC_PER_NODE=8 \
swift rlhf --config projects/baselines/PAPO/configs/PAPO_GRPO_ViRL39K_Qwen25-VL-${MODEL_SIZE_NUM}B.yaml \
    "${EXTRA_ARGS[@]}"
