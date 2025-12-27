import re
import json
import requests
from typing import List, Dict


def magnify_wsi(inputs: List[Dict], target_url='http://127.0.0.1:18888/magnifywsi'):
    #print(f"ZSXM magnify_wsi| {[[i for i in x['images'] if isinstance(i, dict)] for x in inputs]}")
    for input in inputs:
        last_output = input['messages'][-1]['content']
        match = re.search(r'<function name="([a-zA-Z0-9_]+)">\s*(.*?)\s*</function>$', last_output, re.DOTALL)
        if match and sum(m['role'] == 'tool' for m in input['messages']) < 2:  # 限制tool最多在对话中出现2次（之前1次<2，和本次1次）
            # # 删除之前对话的思考过程
            # for mess in input['messages']:
            #     if mess['role'] == 'assistant':
            #         mess['content'] = re.sub(r'<think>.*?</think>', '', mess['content'], flags=re.DOTALL).strip()

            function_name = match.group(1)
            arguments_str = match.group(2)
            try:
                arguments_dict = json.loads(arguments_str)
            except json.JSONDecodeError:
                input['messages'].append({'role': 'tool', 'content': 'The parameters for your function call are not in the correct format. Please use JSON to represent the parameters.'})
                input['finished'] = False
                continue
            
            payload = {
                'function_name': function_name,
                'arguments': arguments_dict,
                'input_images': input['images'] if 'images' in input else [],
            }
            headers = {'Content-Type': 'application/json'}
            response = requests.post(target_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status() # 如果状态码是 4xx 或 5xx 则抛出异常

            try:
                response_data = response.json() # 尝试解析响应体为 JSON
            except json.JSONDecodeError:
                raise

            response_status = response_data.get('status')
            response_content = response_data.get('content')
            if response_status == 'success':
                image_format = response_data.get('image_format', 'png') # Default to png if format missing
                images_base64 = response_data.get('images')
                assert len(images_base64) == response_content.count('<image>'), f"Image count mismatch: {len(images_base64)=} != {response_content.count('<image>')=}"

                for img_base64 in images_base64:
                    image = f'data:image/{image_format};base64,{img_base64}'
                    input['images'].append({'bytes': None, 'path': image})

            input['messages'].append({'role': 'tool', 'content': response_content})
            input['finished'] = False
        else:
            input['finished'] = True
    #print(f"ZSXM magnify_wsi image_num: {[len(x['images']) for x in inputs]}\nroles: {[[m['role'] for m in x['messages']] for x in inputs]}\ninputs: {[x['messages'][1:] for x in inputs]}")
    return inputs
