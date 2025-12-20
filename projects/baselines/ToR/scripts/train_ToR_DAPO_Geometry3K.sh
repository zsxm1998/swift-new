set -x

# 从环境变量提取Bark相关参数
EXTRA_ARGS=()
if [[ -n "$BARK_URL" ]]; then
    EXTRA_ARGS+=(--swanlab_bark_url "$BARK_URL")
fi
if [[ -n "$BARK_DEVICE_TOKEN" ]]; then
    EXTRA_ARGS+=(--swanlab_bark_key "$BARK_DEVICE_TOKEN")
fi

WANDB_PROJECT="ToR-Reproduce" \
MAX_PIXELS=1003520 \
NPROC_PER_NODE=8 \
swift rlhf --config projects/baselines/ToR/configs/ToR_DAPO_Geometry3K.yaml "${EXTRA_ARGS[@]}"