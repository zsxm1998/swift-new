import base64
import imghdr
import mimetypes
import os
from copy import deepcopy
from typing import List, Optional

from openai import OpenAI

from swift.llm.infer.protocol import InferRequest, RequestConfig
from swift.llm.sampling.vanilla_sampler import VanillaSampler
from swift.ray import RayHelper
from .utils import get_messages_md5


class OpenAIEngine:

    def __init__(
        self,
        model: str,
        stream: bool = False,
        base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key: str = '',
        **kwargs,
    ):
        self.model = model
        self.stream = stream
        self.client = OpenAI(api_key=api_key if api_key else os.getenv('OPENAI_API_KEY'), base_url=base_url, **kwargs)

    def infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
    ):
        resp_contents = []
        for infer_request in infer_requests:
            messages = infer_request['messages']
            images = infer_request.get('images') or []
            if images:
                messages = self._build_multimodal_messages(messages, images)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=request_config.temperature,
                top_p=request_config.top_p,
                max_tokens=request_config.max_tokens,
                stream=self.stream,
            )
            reasoning_content = None
            if self.stream:
                reasoning_content = ''
                content = ''
                for chunk in completion:
                    chunk_choices = chunk.choices
                    if len(chunk_choices) == 0:
                        continue
                    reasoning_chunk = chunk_choices[0].delta.reasoning_content if hasattr(
                        chunk_choices[0].delta, 'reasoning_content') else ''
                    answer_chunk = chunk_choices[0].delta.content
                    if reasoning_chunk:
                        reasoning_content += reasoning_chunk
                    elif answer_chunk:
                        content += answer_chunk
            else:
                if hasattr(completion.choices[0].message, 'reasoning_content'):
                    reasoning_content = completion.choices[0].message.reasoning_content
                content = completion.choices[0].message.content
            assert len(content) > 0, 'Empty completion'
            if reasoning_content:
                resp_content = f'<think>{reasoning_content}</think>\n\n<answer>{content}</answer>'
            else:
                resp_content = content
            resp_contents.append(resp_content)

        return resp_contents

    def _build_multimodal_messages(self, messages, images):
        image_iter = iter(images)
        new_messages = []
        for message in messages:
            content = message.get('content')
            if not isinstance(content, str) or '<image>' not in content:
                new_messages.append(message)
                continue
            parts = content.split('<image>')
            mm_content = []
            for i, part in enumerate(parts):
                if part:
                    mm_content.append({'type': 'text', 'text': part.strip()})
                if i < len(parts) - 1:
                    image_path = next(image_iter, None)
                    if not image_path:
                        continue
                    mm_content.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': self._image_path_to_data_url(image_path),
                        },
                    })
            new_message = dict(message)
            new_message['content'] = mm_content
            new_messages.append(new_message)
        return new_messages

    def _image_path_to_data_url(self, image_path: str) -> str:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        mime = mimetypes.guess_type(image_path)[0]
        if not mime:
            kind = imghdr.what(None, h=img_bytes)
            mime = f'image/{kind}' if kind else 'image/png'
        b64 = base64.b64encode(img_bytes).decode('ascii')
        return f'data:{mime};base64,{b64}'


@RayHelper.worker(group=['sampler', 'prm', 'orm'])
class DistillSampler(VanillaSampler):

    def __init__(self, *args, **kwargs):
        super(VanillaSampler, self).__init__(*args, **kwargs)
        assert self.args.sampler_engine == 'client'
        self._prepare_sampler()
        self.caches = self.read_cache()

    @RayHelper.function(group='sampler')
    def _prepare_sampler(self):
        self.infer_engine = OpenAIEngine(model=self.args.model, stream=self.args.stream, **self.args.engine_kwargs)
        self.infer_engine.strict = False

    def _prepare_model_tokenizer(self):
        pass

    def _prepare_template(self):
        pass

    def extract_choice(self, resp):
        message = resp.choices[0].message
        if hasattr(message, 'reasoning_content'):
            reps_content = f'<think>{message.reasoning_content}</think>\n\n<answer>{message.content}</answer>'
        else:
            reps_content = message.content
        return reps_content

    @RayHelper.function(
        group='sampler',
        dispatch=lambda n, i, data:
        ([{
            'messages': data['messages'][i * len(data['messages']) // n:(i + 1) * len(data['messages']) // n]
        }], {}),
        collect='flatten')
    def generate(self, data):
        resp_all = []
        infer_requests = []
        sent = 0
        rows = self.convert_data_to_rows(data)
        for idx, row in enumerate(rows):
            row = deepcopy(row)
            messages = row['messages']
            uuid = get_messages_md5(row)
            if uuid in self.caches:
                choices = self.caches[uuid]['choices']
                if len(choices) == self.args.num_return_sequences:
                    continue
            if self.args.system:
                if messages[0]['role'] == 'system':
                    messages[0]['content'] = self.args.system
                else:
                    messages.insert(0, {'role': 'system', 'content': self.args.system})
            if messages[-1]['role'] == 'assistant':
                messages = messages[:-1]

            row['messages'] = messages
            infer_request = row
            for i in range(self.args.num_return_sequences):
                infer_requests.append(deepcopy(infer_request))
            sent += 1

        request_config = RequestConfig(
            max_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
        )

        resp_list = []
        if len(infer_requests) > 0:
            resp_list = self.infer_engine.infer(infer_requests, request_config=request_config)

        _cur = 0
        for idx, row in enumerate(rows):
            row = deepcopy(row)
            uuid = get_messages_md5(row)
            if uuid in self.caches:
                choices = self.caches[uuid]['choices']
                if len(choices) == self.args.num_return_sequences:
                    row['choices'] = choices
                    resp_all.append(row)
                    continue

            resps = row
            resps['choices'] = []
            for j in range(self.args.num_return_sequences * _cur, self.args.num_return_sequences * (_cur + 1)):
                resps['choices'].append(resp_list[j])
            resp_all.append(resps)
            _cur += 1
        return resp_all
