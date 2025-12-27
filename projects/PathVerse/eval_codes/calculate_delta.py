import argparse
import os
import os.path as osp
import json
from tqdm import tqdm
import re
import copy
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Any, Callable, Dict, Optional, Union

from swift.llm import get_model_tokenizer, to_device, get_template, AutoPreprocessor
from datasets import Dataset as HfDataset


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

def dump_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')

def build_all_choice_prefixes(): # 构造所有可能的选项标记
    # 英文字母
    letter_choices = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + \
                     [chr(i) for i in range(ord('a'), ord('z') + 1)]

    # 数字
    digit_choices = list(map(str, range(1, 100)))

    # 罗马数字
    roman_numerals = [
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
        "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
        "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXVI", "XXVII", "XXVIII", "XXIX", "XXX",
        "XXXI", "XXXII", "XXXIII", "XXXIV", "XXXV", "XXXVI", "XXXVII", "XXXVIII", "XXXIX", "XL",
        "XLI", "XLII", "XLIII", "XLIV", "XLV", "XLVI", "XLVII", "XLVIII", "XLIX", "L"
    ]
    roman_numerals += [r.lower() for r in roman_numerals]

    labels = letter_choices + digit_choices + roman_numerals

    extra_patterns = []
    for label in labels:
        escaped = re.escape(label)
        extra_patterns.extend([
            rf"^\(?{escaped}\)?[\.:)\s]",   # A. A) (A) A:
            rf"^Option\s+{escaped}:",       # Option 1:
            rf"^{escaped}\)$",              # 1)
            rf"^{escaped}$"                 # just the label
        ])
    return labels, extra_patterns


def is_choice_format(s): # 判断是否是选择项
    s = s.strip()
    _, patterns = build_all_choice_prefixes()
    combined = re.compile("|".join(patterns))
    return bool(combined.match(s))


def extract_choice_label(s): # 提取选项标签
    s = s.strip()
    if not is_choice_format(s):
        return False
    label_match = re.match(r"^(?:Option\s+)?\(?([A-Za-z0-9]+)\)?(?=[\.:)\s])?", s)
    return label_match.group(1) if label_match else False

def extract_between_tags(text, start_tag, end_tag, include_tags=False, return_origin=False):
    try:
        start_index = text.index(start_tag)
        end_index = text.index(end_tag, start_index + len(start_tag))
    except ValueError:
        if return_origin:
            return text
        return ""
    
    return text[start_index:end_index + len(end_tag)] if include_tags else text[start_index + len(start_tag):end_index]

def check_negative_exist(text, fragment_list=None):
    if fragment_list is None:
        fragment_list = [
            # 英文否定
            'no.', 'no,', 'no ', 'not ', 'none', 'nobody', 'nothing', 'neither', 'nor ',
            'cannot', "can't", "don't", "doesn't", "didn't", "won't", "wouldn't",
            "isn't", "aren't", "wasn't", "weren't", 'without',
            # 中文否定
            '未', '无', '没', '否', '不', '缺', '排除', '拒绝', '阴性',
        ]

    # 小写处理用于英文匹配
    text_lower = text.lower()
    
    for fragment in fragment_list:
        if fragment in text or fragment in text_lower:
            return True
    return False

def from_jsonl_to_dataset(pivot, jsonl_list):
    pivot_dataset = load_json(pivot)
    for data in pivot_dataset:
        data.pop('task')
    dataset = []
    for jsonl_file in tqdm(jsonl_list):
        jsonl_dataset = load_jsonl(jsonl_file)
        run_id = int(osp.splitext(osp.basename(jsonl_file))[0].split('_')[-1])
        assert len(pivot_dataset) == len(jsonl_dataset), f'{len(pivot_dataset)=} != {len(jsonl_dataset)=}'
        for orig, pred in zip(pivot_dataset, jsonl_dataset):
            assert orig['images'] == pred['images'], f"{orig['images']=} != {pred['images']=}"
            data = copy.deepcopy(orig)
            data['messages'].append({'role':'assistant', 'content': pred['model_response']})
            data['sample_id'] = pred['question_id'] # start from 0
            data['run_id'] = run_id # start from 1

            gt_choice = extract_choice_label(pred['gt_answer'])
            pr_answer = extract_between_tags(pred['model_response'], '<answer>', '</answer>', return_origin=False)
            pr_choice = extract_choice_label(pr_answer)
            correct = 0
            if gt_choice and pr_choice and gt_choice == pr_choice:
                correct = 1
            elif gt_choice == pr_choice == False and check_negative_exist(pred['gt_answer']) and check_negative_exist(pr_answer):
                correct = 1
            data['correct'] = correct
            dataset.append(data)
    autopreprocessor = AutoPreprocessor()
    return autopreprocessor(HfDataset.from_list(dataset))

class LLMDataset(Dataset):
    def __init__(self,
                 dataset: HfDataset,
                 encode_func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self.dataset = dataset
        self.encode_func = encode_func

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.dataset[idx]
        encoded_data = self.encode_func(data)
        encoded_data['sample_id'] = data['sample_id']
        encoded_data['run_id'] = data['run_id']
        encoded_data['correct'] = data['correct']
        return encoded_data

    def __len__(self) -> int:
        return len(self.dataset)

def add_meta(original_collator):
    def new_collator(batch, *, padding_to = None):
        # 调用原始 collator
        res = original_collator(batch, padding_to=padding_to)

        # 提取 sample_id, run_id, correct
        res['sample_id'] = [x['sample_id'] for x in batch]
        res['run_id'] = [x['run_id'] for x in batch]
        res['correct'] = [x['correct'] for x in batch]
        return res

    return new_collator

def compute_loss_per_batch(logits, targets, ignore_index=-100):
    b, s, c = logits.shape
    logits_flat = logits.view(-1, c)
    targets_flat = targets.view(-1)
    loss_flat = torch.nn.functional.cross_entropy(logits_flat, targets_flat, reduction='none', ignore_index=ignore_index)
    loss = loss_flat.view(b, s)
    mask = (targets != ignore_index).float()
    return (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

def remove_images_and_tokens(example):
    if 'images' in example:
        example.pop('images')
    for m in example['messages']:
        if m['role'] in ['user', 'tool'] and '<image>' in m['content']:
            m['content'] = m['content'].replace('<image>', '').strip()
    return example


def main(args):
    model, processor = get_model_tokenizer(args.model)
    template = get_template(model.model_meta.template, processor, max_pixels=1280*28*28)
    template.set_mode('train')
    if template.use_model:
        template.model = model

    dataset = from_jsonl_to_dataset(args.pivot, args.jsonl)
    noimage_dataset = copy.deepcopy(dataset)
    noimage_dataset = noimage_dataset.map(remove_images_and_tokens)
    dataset = LLMDataset(dataset, encode_func=template.encode)
    noimage_dataset = LLMDataset(noimage_dataset, encode_func=template.encode)

    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        collate_fn=add_meta(template.data_collator),
        num_workers=args.bs,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    noimage_dataloader = DataLoader(
        noimage_dataset,
        batch_size=args.bs,
        collate_fn=add_meta(template.data_collator),
        num_workers=args.bs,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    result_list = []
    for withimage_batch, noimage_batch in tqdm(zip(dataloader, noimage_dataloader), total=len(dataloader)):
        assert withimage_batch['sample_id'] == noimage_batch['sample_id'] and withimage_batch['run_id'] == noimage_batch['run_id']
        batch_results = []
        for a, b, c in zip(noimage_batch['sample_id'], noimage_batch['run_id'], noimage_batch['correct']):
            batch_results.append({'sample_id':a, 'run_id':b, 'correct':c})
        withimage_batch.pop('sample_id'), withimage_batch.pop('run_id'), withimage_batch.pop('correct')
        noimage_batch.pop('sample_id'), noimage_batch.pop('run_id'), noimage_batch.pop('correct')
        
        with torch.inference_mode():
            withimage_batch = to_device(withimage_batch, model.device)
            withimage_labels = withimage_batch.pop('labels')
            withimage_output = model(**withimage_batch)
            noimage_batch = to_device(noimage_batch, model.device)
            noimage_labels = noimage_batch.pop('labels')
            noimage_output = model(**noimage_batch)
            
            withimage_loss = compute_loss_per_batch(
                withimage_output.logits[..., :-1, :].contiguous(),
                withimage_labels[..., 1:].contiguous()
            ).tolist()
            noimage_loss = compute_loss_per_batch(
                noimage_output.logits[..., :-1, :].contiguous(),
                noimage_labels[..., 1:].contiguous()
            ).tolist()

        for i, (with_image_ce, no_image_ce) in enumerate(zip(withimage_loss, noimage_loss)):
            batch_results[i]['with_image_ce'] = with_image_ce
            batch_results[i]['no_image_ce'] = no_image_ce
            batch_results[i]['delta'] = no_image_ce - with_image_ce
            batch_results[i]['ratio'] = 1 - (with_image_ce+1e-3) / (no_image_ce+1e-3)
            
        result_list.extend(batch_results)
    dump_jsonl(result_list, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Checkpoint path of the model')
    parser.add_argument('--pivot', type=str, required=True, help='Path to pivot JSON file')
    parser.add_argument('--jsonl', type=str, nargs='+', required=True, help='List of jsonl file paths')
    parser.add_argument('--bs', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--output', type=str, required=True, help='Output path for saving result JSONL')
    args = parser.parse_args()
    main(args)
