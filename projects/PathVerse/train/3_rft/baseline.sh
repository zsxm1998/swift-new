#!/bin/bash
set -euo pipefail
set -x

# -------------------------
# 1) 解析参数（支持短参/长参）
# -------------------------
CONFIG_PATH=""
GPU_LIST=""
RESUME_FROM_CHECKPOINT=""

print_usage() {
  echo "Usage: $0 -c|--config <yaml_path> -g|--gpu <gpu_indices> [-r|--resume_from_checkpoint <ckpt_path>]"
  exit 1
}

# 手动解析 long opts（bash 原生 getopts 不直接支持长参，这里用 while+case）
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      [[ $# -ge 2 ]] || print_usage
      CONFIG_PATH="$2"
      shift 2
      ;;
    -g|--gpu)
      [[ $# -ge 2 ]] || print_usage
      GPU_LIST="$2"
      shift 2
      ;;
    -r|--resume_from_checkpoint)
      [[ $# -ge 2 ]] || print_usage
      RESUME_FROM_CHECKPOINT="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      ;;
    *)
      echo "未知参数：$1"
      print_usage
      ;;
  esac
done

# 必填校验
if [[ -z "${CONFIG_PATH}" || -z "${GPU_LIST}" ]]; then
  echo "错误：-c/--config, -g/--gpu 均为必填参数"
  print_usage
fi

# -------------------------
# 2) 基础校验：config 文件是否存在
# -------------------------
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "错误：config 文件不存在：${CONFIG_PATH}"
  exit 2
fi

# -------------------------
# 4) 解析 GPU 列表，设置 CUDA_VISIBLE_DEVICES / NPROC_PER_NODE
# -------------------------
# 简单合法性：只允许数字和逗号
if [[ ! "${GPU_LIST}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "错误：--gpu 格式不合法，应为 0 或 0,1,2,3 这种形式。当前：${GPU_LIST}"
  exit 4
fi

# 统计 GPU 数量
IFS=',' read -r -a GPU_ARR <<< "${GPU_LIST}"
GPU_COUNT="${#GPU_ARR[@]}"

# -------------------------
# 3) 从环境变量提取 Bark 相关参数
# -------------------------
EXTRA_ARGS=()
if [[ -n "${BARK_URL:-}" ]]; then
  EXTRA_ARGS+=(--swanlab_bark_url "${BARK_URL}")
fi
if [[ -n "${BARK_DEVICE_TOKEN:-}" ]]; then
  EXTRA_ARGS+=(--swanlab_bark_key "${BARK_DEVICE_TOKEN}")
fi

# -------------------------
# 4) 可选：resume_from_checkpoint
#   - 当传入 -r/--resume_from_checkpoint 时：
#     检查 SWANLAB_RESUME 和 SWANLAB_RUN_ID 是否都已设置，否则报错退出
#     然后将 --resume_from_checkpoint 追加进 EXTRA_ARGS
# -------------------------
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  if [[ -z "${SWANLAB_RESUME:-}" || -z "${SWANLAB_RUN_ID:-}" ]]; then
    echo "错误：传入 --resume_from_checkpoint 时，必须同时设置环境变量 SWANLAB_RESUME 和 SWANLAB_RUN_ID。"
    echo "  当前 SWANLAB_RESUME='${SWANLAB_RESUME:-}'"
    echo "  当前 SWANLAB_RUN_ID='${SWANLAB_RUN_ID:-}'"
    exit 6
  fi
  EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

# -------------------------
# 5) 执行 swift rlhf
# 1003520 = 1280*28*28; 200704 = 256*28*28
# MAX_PIXELS是Qwen2-VL的，IMAGE_MAX_TOKEN_NUM是Qwen3-VL的
# -------------------------
CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
NPROC_PER_NODE="${GPU_COUNT}" \
MASTER_PORT=$((RANDOM%10000+20000)) \
MAX_PIXELS=1003520 \
IMAGE_MAX_TOKEN_NUM=1280 \
swift rlhf --config "${CONFIG_PATH}" \
  --max_pixels 1003520 \
  "${EXTRA_ARGS[@]}"
