import os.path as osp
import json
import argparse
import inspect
from tqdm import tqdm
from typing import List, Dict
from dataclasses import asdict
from copy import deepcopy

from swift.llm import InferClient, VllmEngine, PtEngine, InferRequest, RequestConfig, AdapterRequest, get_model_info_meta
from swift.llm.infer.protocol import ChatCompletionResponse

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from origin_multi_turn import magnify_wsi

# from swift.utils import import_external_file
#import_external_file('zsxm_model/models/omnipt_qwen2_vl/swift_register_omnipt_qwen2_vl.py')
#import_external_file('zsxm_model/models/PathVerse/swift_register_pathverse.py')


def load_json(path):
    with open(path, 'r') as jf:
        data = json.load(jf)
    return data


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def preprocess_data(args):
    question_file, image_folder = args.question_file, args.image_folder
    if question_file.endswith(".json"):
        questions = load_json(question_file)
        for i, data in enumerate(questions):
            if 'image' in data:
                data['images'] = data.pop('image')
            if 'images' in data and isinstance(data['images'], str):
                data['images'] = [data['images']]
            if data['messages'][-1]['role'] in ['assistant', 'gpt', 'bot']:
                data['answer'] = data['messages'].pop()['content']
            elif 'solution' in data:
                data['answer'] = data.pop('solution')
            data['question_id'] = data.get('question_id', i)
    else:
        ori_questions = load_jsonl(question_file)
        questions = []
        for i, data in enumerate(ori_questions):
            qdata = {}
            if 'image' in data:
                qdata['images'] = data['image']
            if 'images' in data:
                qdata['images'] = data['images']
            if 'images' in qdata and isinstance(qdata['images'], str):
                qdata['images'] = [qdata['images']]
            qdata['messages'] = [{'role': 'user', 'content': data['text']}]
            if 'answer' in data:
                qdata['answer'] = data['answer']
            qdata['question_id'] = data.get('question_id', i)
            questions.append(qdata)
    for data in questions:
        if 'images' in data:
            for i, image in enumerate(data['images']):
                if not image.startswith("http") and not osp.exists(image):
                    if image_folder and not osp.isabs(image):
                        data['images'][i] = osp.join(image_folder, image)
                    else:
                        raise ValueError(f'Image "{image}" not found.')
            user_text = '\n'.join([x['content'] for x in data['messages'] if x['role'] in ['user', 'human', 'tool']])
            user_image_count = user_text.count('<image>')
            if user_image_count < len(data['images']):
                data['messages'][-1]['content'] = ''.join(['<image>\n']*(len(data['images'])-user_image_count)) + data['messages'][-1]['content']
        if data['messages'][0]['role'] not in ['system', 'system_prompt'] or args.ignore_origin_system:
            system_str = args.system or ''
            if args.think:
                system_str = system_str + '\n\n' + args.think_system if system_str else args.think_system
            if args.func:
                system_str = system_str + '\n\n' + args.func_system if system_str else args.func_system
            if system_str:
                if data['messages'][0]['role'] not in ['system', 'system_prompt']:
                    data['messages'].insert(0, {'role': 'system', 'content': system_str})
                else:
                    data['messages'][0]['content'] = system_str
    return questions

def main(args):
    # 处理数据，将不同格式的输入进行统一，并处理图片路径问题，返回messages格式的数据
    dataset = preprocess_data(args)

    # 根据条件创建推理后端或确定vllm后端
    if args.model_path: # 使用模型创建vllm推理后端
        try:
            engine_kwargs = {
                'compilation_config': {"mode": "none"}
            } # for error torch._dynamo.exc.Unsupported: non-function or method super: <built-in function _disabled_torch_function_impl>
            if args.seed is not None:
                engine_kwargs['seed'] = args.seed
            engine = VllmEngine(
                model_id_or_path=args.model_path,
                gpu_memory_utilization=0.8,
                tensor_parallel_size=args.tensor_parallel_size,
                enable_lora=args.lora_path is not None,
                max_lora_rank=16,
                use_async_engine=False,
                max_num_seqs=args.batch_size if args.batch_size > 1 else inspect.signature(VllmEngine.__init__).parameters['max_num_seqs'].default,
                limit_mm_per_prompt={"image": 10, "video": 1},
                engine_kwargs=engine_kwargs,
                #enforce_eager=True, # for error torch._dynamo.exc.Unsupported: non-function or method super: <built-in function _disabled_torch_function_impl>
            )
        except Exception as e:
            print(f"Failed to initialize VllmEngine: {e}")
            args.batch_size = 2 if get_model_info_meta(args.model_path)[1].model_type in ['pathverse', 'omnipt_qwen2_vl'] or args.func else args.batch_size // 2
            engine = PtEngine(
                model_id_or_path=args.model_path,
                max_batch_size=args.batch_size,
                attn_impl='flash_attn',
                use_hf=True,
            )
            # 如果遇到了timm模型加载404的情况，优先考虑环境变量VLLM_USE_MODELSCOPE设为True导致vllm把modelscope patch给了huggingface
            print("Fallback to PtEngine due to VllmEngine initialization failure.")
    else: # 使用现成的vllm后端
        engine = InferClient(host=args.vllm_host, port=args.vllm_port, timeout=3600)
        vllm_models = engine.models
        if args.vllm_model and args.vllm_model not in vllm_models:
            raise ValueError(f'Model "{args.vllm_model}" not found in vllm models: {vllm_models}')
        elif not args.vllm_model:
            args.vllm_model = vllm_models[0]
        if args.batch_size > 1:
            engine.llm_max_batch_size = args.batch_size
            engine.mllm_max_batch_size = args.batch_size

    # 准备推理
    infer_kwargs = {'request_config': RequestConfig(
        max_tokens = args.max_tokens,
        temperature = args.temperature,
        top_k = (-1 if isinstance(engine, (VllmEngine, InferClient)) else 50) if args.seed is not None else None,
        top_p = 1 if args.seed is not None else None,
        repetition_penalty = 1 if args.seed is not None else None,
        seed = args.seed
    )} # 这里对top_k/top_p/repetition_penalty这样处理是为了在不设置seed的情况下和原来代码保持表现相同
    infer_kwargs['use_tqdm'] = False #True if args.batch_size > 1 else False
    if args.lora_path:
        infer_kwargs['adapter_request'] = AdapterRequest('lora1', args.lora_path)
    if args.vllm_model:
        infer_kwargs['model'] = args.vllm_model

    # 推理
    ans_file = open(args.answers_file, "w")
    if args.batch_size > 1:
        for i in tqdm(range(0, len(dataset), args.batch_size)):
            inputs_slice = [InferRequest(messages=data['messages'], images=data['images']) for data in dataset[i:i + args.batch_size]]
            resp_list: List[ChatCompletionResponse] = engine.infer(inputs_slice, **infer_kwargs)
            if args.func: # 处理函数调用导致的多轮对话
                messages_list = [None] * args.batch_size
                remove_response = True # 这个flag控制从第二轮开始如果messages最后是assistant则concat回复内容
                while len(inputs_slice) > 0:
                    inputs = []
                    for j, output in enumerate(resp_list):
                        choice = output.choices[0]
                        _input: Dict = asdict(inputs_slice[j]) if isinstance(inputs_slice[j], InferRequest) else deepcopy(inputs_slice[j])
                        if remove_response or _input['messages'][-1]['role'] != 'assistant' or not \
                                _input['messages'][-1]['content']:
                            InferRequest.remove_response(_input['messages'])
                            _input['messages'].append({'role': 'assistant', 'content': choice.message.content})
                        else:
                            _input['messages'][-1]['content'] += choice.message.content
                        if 'index' not in _input:
                            _input['index'] = j
                        _input['finish_reason'] = choice.finish_reason
                        inputs.append(_input)
                    results: List[Dict] = magnify_wsi(inputs)
                    inputs_slice = [r for r in results if not r['finished']]
                    for r in results:
                        if r['finished'] or r['finish_reason'] == 'length':
                            messages_list[r['index']] = (r['messages'], r['finish_reason'])
                    if len(inputs_slice) > 0:
                        _input_std = [InferRequest.from_dict(_input) for _input in inputs_slice]
                        resp_list = engine.infer(infer_requests=_input_std, **infer_kwargs)
                    remove_response = False # concat responses from the second loop
                
                for (mess, finish_reason), data in zip(messages_list, dataset[i:i + args.batch_size]):
                    if mess[0]['role'] == 'system':
                        mess.pop(0)  # Remove system message if exists
                    response = mess[-1]['content']
                    prompt_idx_list = [j for j, m in enumerate(data['messages']) if m['role'] == 'user']
                    ans_file.write(json.dumps({
                        "question_id": data['question_id'],
                        "images": data['images'],
                        "prompt": data['messages'][prompt_idx_list[-1 if args.prompt_last else 0]]['content'],
                        "model_response": response,
                        "gt_answer": data.get('answer', None),
                        "messages": mess,
                        "finish_reason": finish_reason,
                    }, ensure_ascii=False) + "\n")
                    ans_file.flush()
            else:
                for resp, data in zip(resp_list, dataset[i:i + args.batch_size]):
                    response = resp.choices[0].message.content
                    prompt_idx_list = [j for j, m in enumerate(data['messages']) if m['role'] == 'user']
                    ans_file.write(json.dumps({
                        "question_id": data['question_id'],
                        "images": data['images'],
                        "prompt": data['messages'][prompt_idx_list[-1 if args.prompt_last else 0]]['content'],
                        "model_response": response,
                        "gt_answer": data.get('answer', None),
                    }, ensure_ascii=False) + "\n")
                    ans_file.flush()
    else:
        # if args.func:
        #     raise NotImplementedError("Function calling mode is not supported for batch size 1.")
        for data in tqdm(dataset):
            infer_requests = [InferRequest(messages=data['messages'], images=data['images'])]
            response: ChatCompletionResponse = engine.infer(infer_requests, **infer_kwargs)[0]
            if args.func:
                remove_response = True
                while True:
                    choice = response.choices[0]
                    _input: Dict = asdict(infer_requests[0]) if isinstance(infer_requests[0], InferRequest) else deepcopy(infer_requests[0])
                    if remove_response or _input['messages'][-1]['role'] != 'assistant' or not \
                            _input['messages'][-1]['content']:
                        InferRequest.remove_response(_input['messages'])
                        _input['messages'].append({'role': 'assistant', 'content': choice.message.content})
                    else:
                        _input['messages'][-1]['content'] += choice.message.content
                    _input['finish_reason'] = choice.finish_reason
                    inputs = [_input]
                    inputs: List[Dict] = magnify_wsi(inputs)
                    if inputs[0]['finished'] or inputs[0]['finish_reason'] == 'length':
                        mess = inputs[0]['messages']
                        if mess[0]['role'] == 'system':
                            mess.pop(0)
                        response = mess[-1]['content']
                        prompt_idx_list = [j for j, m in enumerate(data['messages']) if m['role'] == 'user']
                        ans_file.write(json.dumps({
                            "question_id": data['question_id'],
                            "images": data['images'],
                            "prompt": data['messages'][prompt_idx_list[-1 if args.prompt_last else 0]]['content'],
                            "model_response": response,
                            "gt_answer": data.get('answer', None),
                            "messages": mess,
                            "finish_reason": finish_reason,
                        }, ensure_ascii=False) + "\n")
                        ans_file.flush()
                        break
                    else:
                        response = engine.infer([InferRequest.from_dict(inputs[0])], **infer_kwargs)[0]
                        remove_response = False
            else:
                response = response.choices[0].message.content
                prompt_idx_list = [j for j, m in enumerate(data['messages']) if m['role'] == 'user']
                ans_file.write(json.dumps({
                    "question_id": data['question_id'],
                    "images": data['images'],
                    "prompt": data['messages'][prompt_idx_list[-1 if args.prompt_last else 0]]['content'],
                    "model_response": response,
                    "gt_answer": data.get('answer', None),
                }, ensure_ascii=False) + "\n")
                ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据相关参数
    parser.add_argument("--question-file", type=str, required=True, help="问题文件路径")
    parser.add_argument("--answers-file", type=str, required=True, help="答案文件路径")
    parser.add_argument("--image-folder", type=str, default=None, help="图片文件夹路径，当图片不是绝对路径时使用")
    parser.add_argument("--prompt-last", action='store_true', help="问题是否放在对话的最后一条消息中，默认为False")

    # 如果使用模型推理，则需要在脚本内部署后端
    parser.add_argument("--model-path", type=str, default=None, help="模型路径")
    parser.add_argument("--lora-path", type=str, default=None, help="LoRA 权重路径，可选")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="张量并行大小")

    # 使用现成的vllm客户端推理，不需要传模型参数，只需要传vllm接口
    parser.add_argument("--vllm-host", type=str, default='127.0.0.1', help="vllm接口地址")
    parser.add_argument("--vllm-port", type=int, default=8000, help="vllm接口端口")
    parser.add_argument("--vllm-model", type=str, default=None, help="vllm模型名称")

    # 推理相关参数
    parser.add_argument("--batch-size", type=int, default=64, help="批处理大小")
    parser.add_argument("--max-tokens", type=int, default=4096, help="最大token数，None则为 max_model_len - num_tokens")
    parser.add_argument("--temperature", type=float, default=1, help="温度参数，取值0~2，0表示贪心搜索，越大越随机")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，默认为None。为None时生成参数使用默认设置，因此温度可能失效导致完全确定性采样。给seed设置值之后会将生成参数的top_k和top_p设置为-1和1，以使温度能够控制随机采样结果。")
    parser.add_argument("--ignore-origin-system", action='store_true', help="设为True则忽略数据集中原始的system，默认为False")
    parser.add_argument("--system", type=str, default="你是由浙江大学VIPA实验室开发的病理多模态智能助手，用于辅助病理医生进行专业、准确、高效的诊断。你能够根据用户输入的图片和文字指令或问题，给出相应的回答。", help="系统提示词，传入则设置为该值")
    parser.add_argument("--think", action='store_true', help="是否使用思考模式，传入则设置为True")
    parser.add_argument("--think-system", type=str, default="在给出回答之前，你**必须**先在心里进行思考。将思考内容放在<think>和</think>之间，在</think>之后进行正式回答。对于有确定性答案的问题，如选择题和填空题等，在回答的最后，将选项或答案放在<answer>和</answer>之间。例如：<think>思考内容</think>正式回答内容和分析<answer>选项或回答短语</answer>", help="思考模式下的额外系统提示词，传入则设置为该值")
    parser.add_argument("--func", action='store_true', help="是否使用函数调用模式，传入则设置为True")
    parser.add_argument("--func-system", type=str, default="下面是你可以调用的工具函数列表：\n```\n[{\n    \"name\": \"get_highres_by_point\",\n    \"description\": \"基于WSI缩略图中的选定坐标获取以该点为中心的固定大小高倍率图像，以完成缩略图无法完成的任务。\",\n    \"parameters\": {\n        \"wsi_index\": {\n            \"type\": \"integer\",\n            \"description\": \"用户提供的WSI缩略图索引，从0开始。\",\n            \"required\": true\n        },\n        \"point_list\": {\n            \"type\": \"array\",\n            \"description\": \"需要放大的点列表。每个点表示为[x, y]，其中x和y为归一化坐标，所有坐标均为0-1000之间的整数。\",\n            \"items\": {\n                \"type\": \"array\",\n                \"description\": \"一个放大点，格式为[x, y]，所有值均为0-1000之间的整数。\",\n                \"minItems\": 2,\n                \"maxItems\": 2,\n                \"items\": {\n                    \"type\": \"integer\",\n                    \"minimum\": 0,\n                    \"maximum\": 1000,\n                    \"examples\": [600, 700]\n                }\n            },\n            \"required\": true\n        }\n    },\n    \"returns\": {\n        \"type\": \"images\",\n        \"description\": \"返回对应坐标点中心的固定大小高倍率图像列表。\"\n    }\n}]\n```\n\n说明：\n- 调用上述工具函数来应对需要进一步处理或包含额外信息检索的用户请求。\n- 如果选择调用函数，仅能按照以下格式回复：\n```\n<function name=\"{function_name}\">{parameters}</function>\n```\n其中：parameters => 一个JSON字典，键为函数参数名，值为相应的参数值。\n\n以下是调用示例：\n```\n<function name=\"example_function_name\">{\"argument_name\": argument_value}</function>\n```\n\n提醒：\n- 函数调用必须遵循指定格式。\n- 必须提供所有必需参数。\n- 每次只能调用一个函数。\n- 整个函数调用必须放在一行内。\n- 函数调用必须放在回复的最后，后面不能有任何其他内容，即整条回复以`</function>`结尾。\n- 如果使用搜索结果回答用户请求，必须添加来源信息。", help="函数调用模式下的额外系统提示词，传入则设置为该值")

    args = parser.parse_args()
    main(args)
