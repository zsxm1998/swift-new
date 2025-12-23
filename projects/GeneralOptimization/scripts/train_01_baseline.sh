#!/bin/bash
set -euo pipefail
set -x

############################################
# 用法：
#   ./run.sh -c projects/GeneralOptimization/configs/01_baseline/01-GRPO_Geometry3K_4-GPU.yaml \
#            -m Qwen/Qwen3-VL-4B-Instruct \
#            -g 0,1,2,3
#
# 参数：
#   -c, --config                 配置文件路径（必须）
#   -m, --model                  HF模型名（必须），如 Qwen/Qwen3-VL-4B-Instruct
#   -g, --gpu                    GPU index（必须），如 0 或 0,1,2,3
#   -r, --resume_from_checkpoint 从 checkpoint 恢复（可选）
############################################

# -------------------------
# 1) 解析参数（支持短参/长参）
# -------------------------
CONFIG_PATH=""
MODEL_ID=""
GPU_LIST=""
RESUME_FROM_CHECKPOINT=""

print_usage() {
  echo "Usage: $0 -c|--config <yaml_path> -m|--model <hf_model_id> -g|--gpu <gpu_indices> [-r|--resume_from_checkpoint <ckpt_path>]"
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
    -m|--model)
      [[ $# -ge 2 ]] || print_usage
      MODEL_ID="$2"
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
if [[ -z "${CONFIG_PATH}" || -z "${MODEL_ID}" || -z "${GPU_LIST}" ]]; then
  echo "错误：-c/--config, -m/--model, -g/--gpu 均为必填参数"
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
# 3) 从 config 路径解析 folder / 自定义内容 / N
#    期望格式：
#    projects/GeneralOptimization/configs/[folder]/[custom]_N-GPU.yaml
# -------------------------
# 解析 folder：configs 之后的一级目录
# 例如：projects/GeneralOptimization/configs/01_baseline/xxx.yaml -> 01_baseline
CONFIG_DIR="$(dirname "${CONFIG_PATH}")"
FOLDER="$(basename "${CONFIG_DIR}")"

# config 文件名（不含路径）
CONFIG_FILE="$(basename "${CONFIG_PATH}")"

# 解析 N：从文件名里匹配 _N-GPU.yaml（N 为整数）
# 例如：01-GRPO_Geometry3K_4-GPU.yaml -> N=4
if [[ "${CONFIG_FILE}" =~ _([0-9]+)-GPU\.yaml$ ]]; then
  N_FROM_CONFIG="${BASH_REMATCH[1]}"
else
  echo "错误：config 文件名不符合期望格式：...[custom]_N-GPU.yaml"
  echo "当前文件名：${CONFIG_FILE}"
  exit 3
fi

# 解析自定义内容 custom：去掉末尾 _N-GPU.yaml
# 注意 custom 可能包含多个 '_' 分隔元素，这里整体保留
CUSTOM_PART="${CONFIG_FILE%_${N_FROM_CONFIG}-GPU.yaml}"

# -------------------------
# 4) 解析 GPU 列表，设置 CUDA_VISIBLE_DEVICES / NPROC_PER_NODE
#    并验证 gpu 数量 == N_FROM_CONFIG
# -------------------------
# 简单合法性：只允许数字和逗号
if [[ ! "${GPU_LIST}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "错误：--gpu 格式不合法，应为 0 或 0,1,2,3 这种形式。当前：${GPU_LIST}"
  exit 4
fi

# 统计 GPU 数量
IFS=',' read -r -a GPU_ARR <<< "${GPU_LIST}"
GPU_COUNT="${#GPU_ARR[@]}"

if [[ "${GPU_COUNT}" -ne "${N_FROM_CONFIG}" ]]; then
  echo "错误：GPU数量(${GPU_COUNT}) 与 config 文件名中的 N(${N_FROM_CONFIG}) 不一致，拒绝运行。"
  echo "  --gpu   = ${GPU_LIST}"
  echo "  config  = ${CONFIG_FILE}"
  exit 5
fi

# -------------------------
# 5) 生成 output_dir：
#    outputs/GeneralOptimization/[folder]/[custom]_[model_name]
#    model_name 取 MODEL_ID 按 / 分割的后半部分
# -------------------------
MODEL_NAME="${MODEL_ID##*/}"
OUTPUT_DIR="outputs/GeneralOptimization/${FOLDER}/${CUSTOM_PART}_${MODEL_NAME}"

# -------------------------
# 6) 从环境变量提取 Bark 相关参数
# -------------------------
EXTRA_ARGS=()
if [[ -n "${BARK_URL:-}" ]]; then
  EXTRA_ARGS+=(--swanlab_bark_url "${BARK_URL}")
fi
if [[ -n "${BARK_DEVICE_TOKEN:-}" ]]; then
  EXTRA_ARGS+=(--swanlab_bark_key "${BARK_DEVICE_TOKEN}")
fi

# -------------------------
# 7) 可选：resume_from_checkpoint
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
# 8) 打印最终信息
# -------------------------
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "MODEL_ID=${MODEL_ID}"
echo "GPU_LIST=${GPU_LIST} (count=${GPU_COUNT}, expected=${N_FROM_CONFIG})"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
echo "EXTRA_ARGS=${EXTRA_ARGS[*]}"

# -------------------------
# 9) 执行 swift rlhf（将 --config/--model/output_dir 自动化）
# 1003520 = 1280*28*28; 200704 = 256*28*28
# MAX_PIXELS是Qwen2-VL的，IMAGE_MAX_TOKEN_NUM是Qwen3-VL的
# -------------------------
CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
NPROC_PER_NODE="${GPU_COUNT}" \
MASTER_PORT=$((RANDOM%10000+20000)) \
MAX_PIXELS=1003520 \
IMAGE_MAX_TOKEN_NUM=1280 \
swift rlhf --config "${CONFIG_PATH}" \
  --model "${MODEL_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_pixels 1003520 \
  "${EXTRA_ARGS[@]}"
