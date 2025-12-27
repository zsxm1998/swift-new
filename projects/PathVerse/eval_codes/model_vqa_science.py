import os
import os.path as osp
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
os.environ['MAX_PIXELS'] = '602112'  # 设置最大图片大小，防止爆显存
import argparse
import torch
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
from swift.llm import PtEngine, VllmEngine, InferRequest, RequestConfig


def load_json(file_path : str):
    with open(file_path, 'r') as f:
        return json.load(f)

# 批量处理相关函数
def split_list(lst, n):
    """将列表分成n份"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # 1. 读取 model_type（可能存在于 args.json）
    model_type = None
    if osp.exists(args.model_path) and osp.isdir(args.model_path):
        args_json = osp.join(args.model_path, "args.json")
        if osp.exists(args_json):
            model_type = load_json(args_json).get("model_type")

    # 2. 判断是否启用 LoRA（lora_path 存在）
    enable_lora_flag = False
    if args.lora_path and osp.exists(args.lora_path):
        enable_lora_flag = True

    # 3. 若是 qwen 模型 或者 启用 LoRA，则使用 PtEngine；否则使用 VllmEngine
    is_qwen_model = (model_type and 'qwen' in model_type.lower())
    if is_qwen_model or enable_lora_flag:
        # 如果启用 LoRA，就在初始化时通过 adapters 一次性载入
        adapters = [args.lora_path] if enable_lora_flag else None
        engine = PtEngine(
            model_id_or_path=args.model_path,
            model_type=model_type,
            adapters=adapters,               # 关键：一次性合并 LoRA
        )
    else:
        engine = VllmEngine(
            model_id_or_path=args.model_path,
            model_type=model_type,
            gpu_memory_utilization=0.9
        )

    # 4. 加载问题数据并分块
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # 5. 创建输出目录并打开文件
    answers_file = osp.expanduser(args.answers_file)
    os.makedirs(osp.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # 6. 配置推理请求
    request_config = RequestConfig(max_tokens=None, temperature=0) # max_tokens=None: max_model_len - num_tokens

    # 7. 逐条推理
    for line in tqdm(questions):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs
        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        message = [{
            'role': 'user',
            'content': [
                {"type": "text", "text": qs},
            ]
        }]
        if 'image' in line:
            message[0]['content'].append({"type": "image", "image": osp.join(args.image_folder, line['image'])})
        data = {'messages': message}
        infer_requests = [InferRequest(**data)]

        resp_list = engine.infer(infer_requests, request_config)
        outputs = resp_list[0].choices[0].message.content.strip()

        # prompt for answer
        if args.answer_prompter:
            outputs_reasoning = outputs

            message = [
                {
                    'role': 'user',
                    'content': [{"type": "text", "text": qs},]
                },
                {
                    'role': 'assistant',
                    'content': outputs_reasoning,
                },
                {
                    'role': 'assistant',
                    'content': '',
                }, # 这一项是为了避免InferRequest的__post_init__将最后的assistant的对话去掉
            ]
            if 'image' in line:
                message[0]['content'].append({"type": "image", "image": osp.join(args.image_folder, line['image'])})
            data = {'messages': message}
            infer_requests = [InferRequest(**data)]

            resp_list = engine.infer(infer_requests, request_config)
            outputs = resp_list[0].choices[0].message.content.strip()
            outputs = outputs_reasoning + '\n The answer is ' + outputs

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": '<image>' + '\n' + cur_prompt if 'image' in line else cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": args.model_path,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")

    # LoRA 路径（可选）
    parser.add_argument("--lora-path", type=str, default=None, help="LoRA 权重路径")

    args = parser.parse_args()

    eval_model(args)
