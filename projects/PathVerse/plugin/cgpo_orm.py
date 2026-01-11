import re
from typing import List
from swift.plugin import ORM, orms
from reward_utils import check_negative_exist, check_other_task_tag_exist, extract_between_tags, extract_choice_label, has_chinese, infer_organ_from_text, det_seg_reward_funcs, choice_rsn_loc_format_reward


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


orms['cgpo_format'] = CGPOFormat
orms['cgpo_accuracy'] = CGPOAccuracy
