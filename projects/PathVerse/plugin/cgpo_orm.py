import io
import math
import os
import re
from typing import List, Optional, Tuple
from swift.plugin import ORM, orms
from swift.utils import JsonlWriter
from reward_utils import check_negative_exist, check_other_task_tag_exist, extract_between_tags, extract_choice_label, has_chinese, infer_organ_from_text, det_no_class_reward, det_seg_reward_funcs, choice_rsn_loc_format_reward, parse_bbox_string
from PIL import Image
import torch
from contextlib import nullcontext
from transformers.integrations import is_deepspeed_zero3_enabled
from swift.utils import get_device, set_device
from swift.utils.utils import disable_deepspeed_zero3


class CGPOFormat(ORM):

    def __init__(self):
        self.task_setags = {
            'seg': ('<contour_list>', '</contour_list>'),
            'det_no_class': ('<bbox_list>', '</bbox_list>'),
            'det_with_class': ('<detection_result>', '</detection_result>')
        }
        self.stag_set = set(token for token, _ in self.task_setags.values())
        self.basic_pattern = r'^<think>.*?</think>.*?<answer>.*?</answer>(?![\s\S])'

    def __call__(self, completions, solution, task, messages, **kwargs) -> List[float]:
        rewards = []
        for content, gt, task_type, msgs in zip(completions, solution, task, messages):
            format_reward = None

            if task_type in ['choice', 'choice_func']:
                mch = re.match(self.basic_pattern, content, re.DOTALL | re.MULTILINE)
                format_reward = 1.0 if mch else 0.0
                if task_type == 'choice_func' and not any(m['role'] == 'tool' for m in msgs) and format_reward > 0.5:
                    format_reward = 0.5
            elif task_type in ['choice_nothink', 'choice_nothink_func']:
                if '<think>' in content or '</think>' in content or '<answer>' in content or '</answer>' in content:
                    format_reward = 0.0
                else:
                    format_reward = 1.0
                if task_type == 'choice_nothink_func' and not any(m['role'] == 'tool' for m in msgs) and format_reward > 0.5:
                    format_reward = 0.5
            elif task_type in ['choice_rsn_loc']:
                format_reward = choice_rsn_loc_format_reward(content, self.basic_pattern)
            elif task_type in ['seg', 'det_no_class', 'det_with_class']:
                stag, etag = self.task_setags.get(task_type, (None, None))
                format_reward = 0.1
                if content.count(stag) == 1 and content.count(etag) == 1 \
                    and content.index(stag) < content.index(etag):
                    format_reward = 1.0
                elif check_negative_exist(content):
                    format_reward = 1.0
                elif check_other_task_tag_exist(content, self.stag_set, stag):
                    format_reward = 0.0
            else:
                raise ValueError(f'Not implement format reward for task "{task_type}"')
            
            assert format_reward is not None
            rewards.append(format_reward)
        
        return rewards


class CGPOAccuracy(ORM):

    def __init__(self):
        self.task_setags = {
            'seg': ('<contour_list>', '</contour_list>'),
            'det_no_class': ('<bbox_list>', '</bbox_list>'),
            'det_with_class': ('<detection_result>', '</detection_result>')
        }
        self.stag_set = set(token for token, _ in self.task_setags.values())
    
    def __call__(self, completions, solution, task, messages, **kwargs) -> List[float]:
        rewards = []
        for content, gt, task_type, msgs in zip(completions, solution, task, messages):
            acc_reward = 0.0

            try:
                if task_type in ['choice', 'choice_func', 'choice_rsn_loc']:
                    # 提取gt和content中的选项标签
                    gt_choice = extract_choice_label(gt) # 选项序号或False
                    content_answer = extract_between_tags(content, '<answer>', '</answer>')
                    content_choice = extract_choice_label(content_answer) # 选项序号或False

                    # 若标签符合则赋予1.0的离散reward
                    is_correct = False
                    if gt_choice and content_choice and gt_choice.lower() == content_choice.lower():
                        acc_reward = 0.9
                        if gt_choice == content_choice:
                            acc_reward = 1.0
                        is_correct = True
                    elif not gt_choice and check_negative_exist(gt) \
                        and not content_choice and check_negative_exist(content_answer):
                        acc_reward = 1.0
                        is_correct = True
                    
                    # 考虑思考语言对reward的影响
                    user_last_query = [m['content'] for m in msgs if m['role'] == 'user'][-1]
                    language_inconsistency = has_chinese(user_last_query) \
                        != has_chinese(content) #has_chinese(extract_between_tags(content, '<think>', '</think>'))
                    # 如果存在语言不一致
                    if language_inconsistency:
                        acc_reward = acc_reward - 0.1

                elif task_type in ['choice_nothink', 'choice_nothink_func']:
                    # 提取gt和content中的选项标签
                    gt_choice = extract_choice_label(gt) # 选项序号或False
                    content_choice = extract_choice_label(content) # 选项序号或False

                    # 若标签符合则赋予1.0的离散reward
                    acc_reward = 0.0
                    if gt_choice and content_choice and gt_choice == content_choice:
                        acc_reward = 1.0
                    elif not gt_choice and check_negative_exist(gt) \
                        and not content_choice and check_negative_exist(content_answer):
                        acc_reward = 1.0

                elif task_type in ['seg', 'det_no_class', 'det_with_class']:
                    stag, etag = self.task_setags.get(task_type, (None, None))
                    reward_func = det_seg_reward_funcs.get(task_type)
                    acc_reward = reward_func(
                        extract_between_tags(gt, stag, etag, include_tags=True, return_origin=True),
                        extract_between_tags(content, stag, etag, include_tags=True, return_origin=True)
                    )

                else:
                    print(f'Not implement acc reward for task "{task_type}"')
                    raise ValueError(f'Not implement acc reward for task "{task_type}"')
            except:
                acc_reward = 0.0
            
            rewards.append(acc_reward)
        
        return rewards


class CGPOEntityDetReward(ORM):

    def __init__(self, agg: str = 'mean', prompt_template: str = '检测图像中的{entity}',
                 prompt_template_en: str = 'Detect the {entity} in the image',
                 log_jsonl_name: str = 'cgpo_entity_det.jsonl', apply_tasks=['choice_rsn_loc']):
        self.agg = agg
        self.prompt_template = prompt_template
        self.prompt_template_en = prompt_template_en
        self.log_jsonl_name = log_jsonl_name
        self.apply_tasks = set(apply_tasks) if apply_tasks else None
        self.ent_pat = re.compile(
            r'<entity\s+name="(?P<name>[^"]+)"\s+id="(?P<id>\d+)">\s*(?P<body>.*?)\s*</entity>',
            re.DOTALL)
        self.bbox_list_pat = re.compile(r'<bbox_list(?:\s+[^>]*)?>(?P<b>.*?)</bbox_list>', re.DOTALL)
        self.bbox_list_class_pat = re.compile(
            r'<bbox_list\s+[^>]*class="(?P<class>[^"]+)"[^>]*>(?P<b>.*?)</bbox_list>',
            re.DOTALL)
        self._jsonl_writer = None
        self._log_fpath = None

    def _aggregate(self, rewards: List[float]) -> float:
        if not rewards:
            return 0.0
        if self.agg == 'max':
            return max(rewards)
        if self.agg == 'min':
            return min(rewards)
        if self.agg == 'mean':
            return sum(rewards) / len(rewards)
        raise ValueError(f'Unsupported agg={self.agg!r}, expected one of: mean, max, min')

    def _extract_entities(self, content: str):
        entities = []
        for m in self.ent_pat.finditer(content):
            name = m.group('name').strip()
            body = m.group('body')
            bbox_m = self.bbox_list_pat.search(body)
            if not bbox_m:
                continue
            bbox_text = f"<bbox_list>{bbox_m.group('b')}</bbox_list>"
            entities.append((name, bbox_text))
        return entities

    def _build_prompt(self, name: str) -> str:
        template = self.prompt_template if has_chinese(name) else self.prompt_template_en
        return '<image>\n' + template.format(entity=name)

    def _extract_bbox_list_text(self, text: str, entity_name: str = None) -> str:
        if entity_name:
            for m in self.bbox_list_class_pat.finditer(text):
                if m.group('class').strip() == entity_name:
                    return f"<bbox_list>{m.group('b')}</bbox_list>"
        bbox_m = self.bbox_list_pat.search(text)
        if bbox_m:
            return f"<bbox_list>{bbox_m.group('b')}</bbox_list>"
        return ""

    def _build_dummy_infer_inputs(self):
        dummy_prompt = '<image>\n这是什么'
        img = Image.new('RGB', (56, 56), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        dummy_image = {'bytes': buf.getvalue(), 'path': None}
        dummy_input = {
            'messages': [{'role': 'user', 'content': dummy_prompt}],
            'images': [dummy_image],
        }
        return [dummy_input]

    def _get_jsonl_writer(self, output_dir: str):
        if not output_dir:
            return None
        fpath = os.path.abspath(os.path.expanduser(os.path.join(output_dir, self.log_jsonl_name)))
        if self._jsonl_writer is None or self._log_fpath != fpath:
            self._jsonl_writer = JsonlWriter(fpath, strict=False)
            self._log_fpath = fpath
        return self._jsonl_writer

    def __call__(self, completions, solution, task, messages, **kwargs) -> List[float]:
        rollout_infer = kwargs.get('rollout_infer')
        images = kwargs.get('images')
        images_list = images if images is not None else [None] * len(completions)

        infer_inputs = []
        infer_meta = []
        per_sample_entity_counts = [0] * len(completions)
        per_sample_entities = [[] for _ in completions]
        per_sample_rewards = [[] for _ in completions]

        if rollout_infer is not None and images is not None:
            for i, (content, task_type) in enumerate(zip(completions, task)):
                if self.apply_tasks and task_type not in self.apply_tasks:
                    continue
                if images_list[i] is None:
                    continue
                entities = self._extract_entities(content)
                if not entities:
                    continue
                per_sample_entity_counts[i] = len(entities)
                for name, pred_bbox_text in entities:
                    prompt = self._build_prompt(name)
                    ent_info = {'name': name, 'gt_bbox_text': '', 'pred_bbox_text': pred_bbox_text}
                    per_sample_entities[i].append(ent_info)
                    infer_inputs.append({
                        'messages': [{'role': 'user', 'content': prompt}],
                        'images': images_list[i],
                    })
                    infer_meta.append((i, name, ent_info))
        elif rollout_infer is not None and images is None:
            rollout_infer(self._build_dummy_infer_inputs())

        infer_outputs = []
        if rollout_infer is not None:
            if infer_inputs:
                infer_outputs = rollout_infer(infer_inputs)
            else:
                rollout_infer(self._build_dummy_infer_inputs())

        if infer_outputs:
            for output, (sample_idx, ent_name, ent_info) in zip(infer_outputs, infer_meta):
                if not output or 'messages' not in output or not output['messages']:
                    per_sample_rewards[sample_idx].append(0.0)
                    continue
                gt_content = output['messages'][-1]['content'] or ''
                gt_bbox_text = self._extract_bbox_list_text(gt_content, ent_name)
                if not gt_bbox_text:
                    gt_bbox_text = gt_content
                ent_info['gt_bbox_text'] = gt_bbox_text
                try:
                    reward = det_no_class_reward(gt_bbox_text, ent_info['pred_bbox_text'])
                except Exception:
                    reward = 0.0
                per_sample_rewards[sample_idx].append(reward)

        output_dir = kwargs.get('output_dir')
        writer = self._get_jsonl_writer(output_dir)
        trainer_state = kwargs.get('trainer_state')
        step = getattr(trainer_state, 'global_step', None) if trainer_state is not None else None
        prompt_ids = kwargs.get('prompt_id')
        request_ids = kwargs.get('request_id')
        log_rows = []
        for i, entities in enumerate(per_sample_entities):
            if not entities:
                continue
            log_rows.append({
                'step': step,
                'prompt_id': prompt_ids[i] if prompt_ids else None,
                'request_id': request_ids[i] if request_ids else None,
                'images': images_list[i],
                'messages': messages[i],
                'solution': solution[i],
                'entities': entities,
            })
        if writer is not None:
            writer.append(log_rows, gather_obj=True)

        rewards = []
        for i, rewards_i in enumerate(per_sample_rewards):
            if not rewards_i and per_sample_entity_counts[i] > 0:
                rewards.append(0.0)
                continue
            rewards.append(self._aggregate(rewards_i))
        return rewards


class CGPOEntityDetConchReward(ORM):

    def __init__(self,
                 agg: str = 'mean',
                 bbox_agg: str = 'mean',
                 conch_model_name: str = 'conch_ViT-B-16',
                 conch_pretrained: str = '/shared_storage/xzsxm/.cache/huggingface/hub/models--MahmoodLab--conch/snapshots/f9ca9f877171a28ade80228fb195ac5d79003357/pytorch_model.bin',
                 log_jsonl_name: str = 'cgpo_entity_det_conch.jsonl',
                 apply_tasks=['choice_rsn_loc']):
        self.agg = agg
        self.bbox_agg = bbox_agg
        self.conch_model_name = conch_model_name
        self.conch_pretrained = conch_pretrained
        self.log_jsonl_name = log_jsonl_name
        self.apply_tasks = set(apply_tasks) if apply_tasks else None
        self.ent_pat = re.compile(
            r'<entity\s+name="(?P<name>[^"]+)"\s+id="(?P<id>\d+)">\s*(?P<body>.*?)\s*</entity>',
            re.DOTALL)
        self.bbox_list_pat = re.compile(r'<bbox_list(?:\s+[^>]*)?>(?P<b>.*?)</bbox_list>', re.DOTALL)

        set_device() # 兜底
        device = get_device()
        ds_context = disable_deepspeed_zero3() if is_deepspeed_zero3_enabled() else nullcontext()
        with ds_context:
            from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
            model, preprocess = create_model_from_pretrained(self.conch_model_name, self.conch_pretrained, device=device)
            tokenizer = get_tokenizer()

        if hasattr(model, 'eval'):
            model.eval()
        self._conch_model = model
        self._conch_preprocess = preprocess
        self._conch_tokenizer = tokenizer
        self._conch_device = device
        self._jsonl_writer = None
        self._log_fpath = None

    def _aggregate(self, rewards: List[float], agg: Optional[str] = None) -> float:
        if not rewards:
            return 0.0
        agg = agg or self.agg
        if agg == 'max':
            return max(rewards)
        if agg == 'min':
            return min(rewards)
        if agg == 'mean':
            return sum(rewards) / len(rewards)
        raise ValueError(f'Unsupported agg={agg!r}, expected one of: mean, max, min')

    def _extract_entities(self, content: str):
        entities = []
        for m in self.ent_pat.finditer(content):
            name = m.group('name').strip()
            body = m.group('body')
            bbox_m = self.bbox_list_pat.search(body)
            if not bbox_m:
                continue
            bbox_text = f"<bbox_list>{bbox_m.group('b')}</bbox_list>"
            entities.append((name, bbox_text))
        return entities

    def _parse_bbox_list(self, bbox_text: str) -> List[Tuple[int, int, int, int]]:
        if not bbox_text or check_negative_exist(bbox_text):
            return []
        box_texts = re.findall(r'<box>\s*([^<]+?)\s*</box>', bbox_text)
        boxes = []
        for text in box_texts:
            m = re.fullmatch(r'\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*', text)
            if not m:
                continue
            x1, y1, x2, y2 = map(int, m.groups())
            if not all(0 <= v <= 1000 for v in (x1, y1, x2, y2)):
                continue
            if x1 >= x2 or y1 >= y2:
                continue
            boxes.append((x1, y1, x2, y2))
        return boxes

    def _load_first_image(self, image_item):
        if image_item is None:
            return None
        if isinstance(image_item, list):
            if not image_item:
                return None
            image_item = image_item[0]
        if isinstance(image_item, Image.Image):
            return image_item.convert('RGB')
        if isinstance(image_item, dict):
            if image_item.get('bytes'):
                return Image.open(io.BytesIO(image_item['bytes'])).convert('RGB')
            if image_item.get('path'):
                return Image.open(image_item['path']).convert('RGB')
        if isinstance(image_item, str):
            return Image.open(image_item).convert('RGB')
        return None

    def _crop_image(self, image: Image.Image, bbox: Tuple[int, int, int, int]):
        w, h = image.size
        x1, y1, x2, y2 = bbox
        px1 = int(round(x1 / 1000 * w))
        px2 = int(round(x2 / 1000 * w))
        py1 = int(round(y1 / 1000 * h))
        py2 = int(round(y2 / 1000 * h))
        px1 = max(0, min(px1, w))
        px2 = max(0, min(px2, w))
        py1 = max(0, min(py1, h))
        py2 = max(0, min(py2, h))
        if px2 <= px1 or py2 <= py1:
            return None
        return image.crop((px1, py1, px2, py2))

    def _get_jsonl_writer(self, output_dir: str):
        if not output_dir:
            return None
        fpath = os.path.abspath(os.path.expanduser(os.path.join(output_dir, self.log_jsonl_name)))
        if self._jsonl_writer is None or self._log_fpath != fpath:
            self._jsonl_writer = JsonlWriter(fpath, strict=False)
            self._log_fpath = fpath
        return self._jsonl_writer

    @torch.inference_mode()
    def _encode_texts(self, texts: List[str]):
        from conch.open_clip_custom import tokenize
        tokens = tokenize(texts=texts, tokenizer=self._conch_tokenizer).to(self._conch_device)
        text_features = self._conch_model.encode_text(tokens, normalize=True, embed_cls=True)
        return text_features

    @torch.inference_mode()
    def _encode_images(self, images: List[Image.Image]):
        image_tensors = [self._conch_preprocess(img) for img in images]
        image_tensor = torch.stack(image_tensors).to(self._conch_device)
        image_features = self._conch_model.encode_image(image_tensor, normalize=True, proj_contrast=True)
        return image_features

    def __call__(self, completions, solution, task, messages, **kwargs) -> List[float]:
        images = kwargs.get('images')
        images_list = images if images is not None else [None] * len(completions)

        per_sample_rewards = [[] for _ in completions]
        per_sample_entities = [[] for _ in completions]

        for i, (content, task_type) in enumerate(zip(completions, task)):
            if self.apply_tasks and task_type not in self.apply_tasks:
                continue
            image = self._load_first_image(images_list[i])
            if image is None:
                continue
            entities = self._extract_entities(content)
            if not entities:
                continue

            entity_names = []
            entity_bbox_texts = []
            for name, bbox_text in entities:
                entity_names.append(name)
                entity_bbox_texts.append(bbox_text)

            unique_names = []
            name_to_index = {}
            for name in entity_names:
                if name not in name_to_index:
                    name_to_index[name] = len(unique_names)
                    unique_names.append(name)

            text_features = self._encode_texts(unique_names)

            for name, bbox_text in zip(entity_names, entity_bbox_texts):
                bboxes = self._parse_bbox_list(bbox_text)
                crops = []
                for bbox in bboxes:
                    crop = self._crop_image(image, bbox)
                    if crop is not None:
                        crops.append(crop)
                if not crops:
                    per_sample_rewards[i].append(0.0)
                    per_sample_entities[i].append({
                        'name': name,
                        'bbox_text': bbox_text,
                        'bbox_rewards': [],
                        'entity_reward': 0.0,
                    })
                    continue

                image_features = self._encode_images(crops)

                text_feature = text_features[name_to_index[name]]
                sims = (image_features * text_feature).sum(dim=-1).cpu().tolist()
                entity_reward = self._aggregate(sims, agg=self.bbox_agg)
                per_sample_rewards[i].append(entity_reward)
                per_sample_entities[i].append({
                    'name': name,
                    'bbox_text': bbox_text,
                    'bbox_rewards': sims,
                    'entity_reward': entity_reward,
                })

        output_dir = kwargs.get('output_dir')
        writer = self._get_jsonl_writer(output_dir)
        trainer_state = kwargs.get('trainer_state')
        step = getattr(trainer_state, 'global_step', None) if trainer_state is not None else None
        prompt_ids = kwargs.get('prompt_id')
        request_ids = kwargs.get('request_id')
        log_rows = []
        for i, entities in enumerate(per_sample_entities):
            if not entities:
                continue
            log_rows.append({
                'step': step,
                'prompt_id': prompt_ids[i] if prompt_ids else None,
                'request_id': request_ids[i] if request_ids else None,
                'images': images_list[i],
                'messages': messages[i],
                'solution': solution[i],
                'entities': entities,
            })
        if writer is not None:
            writer.append(log_rows, gather_obj=True)

        rewards = []
        for rewards_i in per_sample_rewards:
            rewards.append(self._aggregate(rewards_i))
        return rewards


class CGPODetNoClassAuxReward(ORM):

    def __init__(self,
                 match_algo: str = 'auto',
                 match_threshold: float = 0.1,
                 center_sigma: float = 1.0,
                 size_sigma: float = 1.0,
                 center_weight: float = 0.5,
                 size_weight: float = 0.5):
        self.match_algo = match_algo
        self.match_threshold = match_threshold
        self.center_sigma = max(center_sigma, 1e-6)
        self.size_sigma = max(size_sigma, 1e-6)
        weight_sum = center_weight + size_weight
        if weight_sum <= 0:
            raise ValueError('center_weight + size_weight must be positive')
        self.center_weight = center_weight / weight_sum
        self.size_weight = size_weight / weight_sum
        self.stag, self.etag = ('<bbox_list>', '</bbox_list>')

    def _parse_boxes(self, text: str) -> List[Tuple[int, int, int, int]]:
        if check_negative_exist(text):
            return []
        boxes = parse_bbox_string(text)
        valid = []
        for x1, y1, x2, y2 in boxes:
            if x1 >= x2 or y1 >= y2:
                continue
            valid.append((x1, y1, x2, y2))
        return valid

    def _center_score(self, gt_box, pred_box) -> float:
        gx1, gy1, gx2, gy2 = gt_box
        px1, py1, px2, py2 = pred_box
        gcx = (gx1 + gx2) / 2.0
        gcy = (gy1 + gy2) / 2.0
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        gw = max(gx2 - gx1, 1.0)
        gh = max(gy2 - gy1, 1.0)
        denom = max(gw, gh, 1.0)
        dx = abs(pcx - gcx) / denom
        dy = abs(pcy - gcy) / denom
        return math.exp(-0.5 * ((dx / self.center_sigma) ** 2 + (dy / self.center_sigma) ** 2))

    def _size_score(self, gt_box, pred_box) -> float:
        gx1, gy1, gx2, gy2 = gt_box
        px1, py1, px2, py2 = pred_box
        gw = max(gx2 - gx1, 1.0)
        gh = max(gy2 - gy1, 1.0)
        pw = max(px2 - px1, 1.0)
        ph = max(py2 - py1, 1.0)
        log_w = math.log(pw / gw)
        log_h = math.log(ph / gh)
        return math.exp(-0.5 * ((log_w / self.size_sigma) ** 2 + (log_h / self.size_sigma) ** 2))

    def _pair_score(self, gt_box, pred_box) -> float:
        center_score = self._center_score(gt_box, pred_box)
        size_score = self._size_score(gt_box, pred_box)
        return self.center_weight * center_score + self.size_weight * size_score

    def _match_pairs(self, scores: List[List[float]]) -> List[Tuple[int, int]]:
        if not scores or not scores[0]:
            return []
        n_gt = len(scores)
        n_pred = len(scores[0])
        use_hungarian = self.match_algo in ('auto', 'hungarian')
        if use_hungarian:
            try:
                from scipy.optimize import linear_sum_assignment
                cost = [[-s for s in row] for row in scores]
                row_ind, col_ind = linear_sum_assignment(cost)
                return list(zip(row_ind.tolist(), col_ind.tolist()))
            except Exception:
                if self.match_algo == 'hungarian':
                    pass
        used_gt = set()
        used_pred = set()
        pairs = []
        while len(used_gt) < n_gt and len(used_pred) < n_pred:
            best = None
            best_score = -1.0
            for i in range(n_gt):
                if i in used_gt:
                    continue
                row = scores[i]
                for j in range(n_pred):
                    if j in used_pred:
                        continue
                    s = row[j]
                    if s > best_score:
                        best_score = s
                        best = (i, j)
            if best is None:
                break
            used_gt.add(best[0])
            used_pred.add(best[1])
            pairs.append(best)
        return pairs

    def __call__(self, completions, solution, task, **kwargs) -> List[float]:
        rewards = []
        for content, gt, task_type in zip(completions, solution, task):
            if task_type != 'det_no_class':
                rewards.append(0.0)
                continue
            aux_reward = 0.0
            try:
                gt_text = extract_between_tags(gt, self.stag, self.etag, include_tags=True, return_origin=True)
                pred_text = extract_between_tags(content, self.stag, self.etag, include_tags=True, return_origin=True)
                gt_boxes = self._parse_boxes(gt_text)
                pred_boxes = self._parse_boxes(pred_text)
                if not gt_boxes and not pred_boxes:
                    aux_reward = 1.0
                elif not gt_boxes or not pred_boxes:
                    aux_reward = 0.0
                else:
                    scores = [
                        [self._pair_score(gt_box, pred_box) for pred_box in pred_boxes]
                        for gt_box in gt_boxes
                    ]
                    pairs = self._match_pairs(scores)
                    matched_scores = [
                        scores[i][j] for i, j in pairs
                        if scores[i][j] >= self.match_threshold
                    ]
                    tp = len(matched_scores)
                    if tp > 0:
                        precision = tp / len(pred_boxes) if pred_boxes else 0.0
                        recall = tp / len(gt_boxes) if gt_boxes else 0.0
                        if precision + recall > 0:
                            f1 = 2 * precision * recall / (precision + recall)
                            quality = sum(matched_scores) / tp
                            aux_reward = f1 * quality
            except Exception:
                aux_reward = 0.0
            rewards.append(aux_reward)
        return rewards


orms['cgpo_format'] = CGPOFormat
orms['cgpo_accuracy'] = CGPOAccuracy
orms['cgpo_entity_det'] = CGPOEntityDetReward
orms['cgpo_entity_det_conch'] = CGPOEntityDetConchReward
orms['cgpo_det_no_class_aux'] = CGPODetNoClassAuxReward
