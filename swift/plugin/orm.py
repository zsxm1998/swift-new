import os
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import json
from mathruler.grader import extract_boxed_content

if TYPE_CHECKING:
    from swift.llm import InferRequest


class ORM:
    """Base class for synchronous outcome reward models (ORM).

    Subclasses should implement the __call__ method to compute rewards.

    Example:
        class MyReward(ORM):
            def __call__(self, completions, **kwargs) -> List[float]:
                return [1.0 if len(c) > 100 else 0.0 for c in completions]
    """

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


def extract_choice_label(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    candidates = []
    boxed = extract_boxed_content(text)
    if boxed and boxed != 'None':
        candidates.append((boxed, True))
    candidates.append((text, False))
    for cand, from_boxed in candidates:
        if cand is None:
            continue
        normalized = cand.replace('\n', ' ').strip()
        answer = re.search(
            r'(?i)\b(?:answer|option|choice)\b\s*[:\-]?\s*[\(\[]?\s*([A-H])\s*[\)\]]?\b',
            normalized,
        )
        if answer:
            return answer.group(1).upper()
        leading = re.search(r'^\s*[\(\[]?\s*([A-H])\s*[\)\]]?\s*[\.\)\:]\s+', normalized)
        if leading:
            return leading.group(1).upper()
        if from_boxed:
            latex_text = re.search(r'\\text\{\s*[\(\[]?\s*([A-H])\s*[\)\]]?\s*\}', normalized)
            if latex_text:
                return latex_text.group(1).upper()
            paren = re.search(r'[\(\[]\s*([A-H])\s*[\)\]]', normalized)
            if paren:
                return paren.group(1).upper()
        direct = re.search(r'^\s*[\(\[]?\s*([A-H])\s*[\)\]]?\s*$', normalized)
        if direct:
            return direct.group(1).upper()
    return None


def extract_yes_no(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    candidates = []
    boxed = extract_boxed_content(text)
    if boxed and boxed != 'None':
        candidates.append((boxed, True))
    candidates.append((text, False))
    for cand, from_boxed in candidates:
        if cand is None:
            continue
        normalized = cand.replace('\n', ' ').strip().lower()
        answer = re.search(r'\banswer\b\s*[:\-]?\s*(yes|no)\b', normalized)
        if answer:
            return answer.group(1)
        direct = re.fullmatch(r'(yes|no)', normalized)
        if direct:
            return direct.group(1)
        leading = re.search(r'^\s*(yes|no)\b', normalized)
        if leading:
            return leading.group(1)
        if from_boxed:
            latex_clean = normalized
            latex_clean = re.sub(r'\\(text|mathrm|mathbf|mathit|mathtt)\{([^}]*)\}', r'\2', latex_clean)
            latex_clean = re.sub(r'[{}$]', '', latex_clean)
            latex_clean = re.sub(r'[^a-z]+', ' ', latex_clean).strip()
            if latex_clean in {'yes', 'no'}:
                return latex_clean
            paren = re.search(r'[\(\[]\s*(yes|no)\s*[\)\]]', normalized)
            if paren:
                return paren.group(1)
    return None


def is_yes_no_correct(prediction: str, ground_truth: str) -> Optional[bool]:
    gt = extract_yes_no(ground_truth)
    if not gt:
        return None
    pred = extract_yes_no(prediction)
    return pred == gt


def _normalize_plain_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r'\\(text|mathrm|mathbf|mathit|mathtt)\{([^}]*)\}', r'\2', lowered)
    lowered = re.sub(r'[{}$]', '', lowered)
    lowered = re.sub(r'[^a-z]+', ' ', lowered)
    return re.sub(r'\s+', ' ', lowered).strip()


def _normalize_alnum_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r'\\(text|mathrm|mathbf|mathit|mathtt)\{([^}]*)\}', r'\2', lowered)
    lowered = re.sub(r'\\\s+', ' ', lowered)
    lowered = re.sub(r'[{}$]', '', lowered)
    lowered = re.sub(r'[^a-z0-9]+', ' ', lowered)
    return re.sub(r'\s+', ' ', lowered).strip()


def extract_boxed_plain_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    boxed = extract_boxed_content(text)
    if not boxed or boxed == 'None':
        return None
    normalized = _normalize_plain_text(boxed)
    if not normalized:
        return None
    return normalized


def extract_boxed_alnum_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    boxed = extract_boxed_content(text)
    if not boxed or boxed == 'None':
        return None
    normalized = _normalize_alnum_text(boxed)
    if not normalized:
        return None
    return normalized


def is_simple_text_correct(prediction: str, ground_truth: str) -> Optional[bool]:
    if not ground_truth:
        return None
    if re.search(r'[\\0-9]', ground_truth):
        return None
    if not re.search(r'[A-Za-z]', ground_truth):
        return None
    if not re.fullmatch(r"[A-Za-z\s\.\,\;\:\!\?\'\"\-]+", ground_truth):
        return None
    gt_norm = _normalize_plain_text(ground_truth)
    if not gt_norm:
        return None
    if len(gt_norm.split()) > 6:
        return None
    pred_norm = extract_boxed_plain_text(prediction)
    if not pred_norm:
        return None
    return pred_norm == gt_norm


def is_simple_alnum_text_correct(prediction: str, ground_truth: str) -> Optional[bool]:
    if not ground_truth:
        return None
    if re.search(r'\\', ground_truth):
        return None
    if not re.search(r'[A-Za-z0-9]', ground_truth):
        return None
    if not re.fullmatch(r"[A-Za-z0-9\s\.\,\;\!\?\'\"\-]+", ground_truth):
        return None
    if re.search(r'[:/=+\-*^]', ground_truth):
        return None
    gt_norm = _normalize_alnum_text(ground_truth)
    if not gt_norm:
        return None
    if len(gt_norm.split()) > 6:
        return None
    pred_norm = extract_boxed_alnum_text(prediction)
    if not pred_norm:
        return None
    return pred_norm == gt_norm


def extract_colon_value(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    candidates = []
    boxed = extract_boxed_content(text)
    if boxed and boxed != 'None':
        candidates.append(boxed)
    candidates.append(text)
    def _normalize_colon(val: str) -> str:
        return re.sub(r'\s+', '', val)

    time_pattern = r'([0-9]+\s*:\s*[0-9]+)\s*(a\.?m\.?|p\.?m\.?)?'
    for cand in candidates:
        if cand is None:
            continue
        normalized = cand.replace('\n', ' ').strip()
        answer = re.search(rf'(?i)\banswer\b\s*[:\-]?\s*{time_pattern}\b', normalized)
        if answer:
            suffix = (answer.group(2) or '').replace('.', '').lower()
            return f"{_normalize_colon(answer.group(1))}{suffix}"
        full = re.fullmatch(rf'\s*{time_pattern}\s*', normalized, re.IGNORECASE)
        if full:
            suffix = (full.group(2) or '').replace('.', '').lower()
            return f"{_normalize_colon(full.group(1))}{suffix}"
        if boxed and cand == boxed:
            paren = re.search(rf'[\(\[]\s*{time_pattern}\s*[\)\]]', normalized, re.IGNORECASE)
            if paren:
                suffix = (paren.group(2) or '').replace('.', '').lower()
                return f"{_normalize_colon(paren.group(1))}{suffix}"
    return None


def is_colon_value_correct(prediction: str, ground_truth: str) -> Optional[bool]:
    gt = extract_colon_value(ground_truth)
    if not gt:
        return None
    pred = extract_colon_value(prediction)
    return pred == gt


def is_multiple_choice_correct(prediction: str, ground_truth: str) -> Optional[bool]:
    gt_choice = extract_choice_label(ground_truth)
    if not gt_choice:
        return None
    pred_choice = extract_choice_label(prediction)
    return pred_choice == gt_choice


class AsyncORM:
    """Base class for asynchronous outcome reward models (ORM).

    Use this for reward functions that involve I/O operations (e.g., API calls,
    database queries) that can benefit from async execution.

    Async reward functions are executed in parallel using asyncio.gather,
    which can significantly speed up reward computation when multiple async
    reward functions are used or when the reward function involves network calls.

    Example:
        class MyAsyncReward(AsyncORM):
            async def __call__(self, completions, **kwargs) -> List[float]:
                # Use asyncio.gather for parallel execution of all API calls
                import asyncio
                import aiohttp

                async def score_single(session, text):
                    async with session.post(api_url, json={'text': text}) as resp:
                        result = await resp.json()
                        return result['score']

                async with aiohttp.ClientSession() as session:
                    tasks = [score_single(session, c) for c in completions]
                    rewards = await asyncio.gather(*tasks)
                    return list(rewards)
    """

    async def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        # pattern = r'\\boxed{([^}]*)}'
        # match = re.search(pattern, text)
        # if match:
        #     return match.group(1).strip()
        # else:
        #     return text
        res = extract_boxed_content(text)
        if res == 'None':
            return text
        return res

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if any(expr is None for expr in parsed_exprs):
            return False
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            yes_no_match = is_yes_no_correct(prediction, ground_truth)
            if yes_no_match is not None:
                rewards.append(float(yes_no_match))
                continue
            colon_match = is_colon_value_correct(prediction, ground_truth)
            if colon_match is not None:
                rewards.append(float(colon_match))
                continue
            choice_match = is_multiple_choice_correct(prediction, ground_truth)
            if choice_match is not None:
                rewards.append(float(choice_match))
                continue
            simple_match = is_simple_text_correct(prediction, ground_truth)
            if simple_match is not None:
                rewards.append(float(simple_match))
                continue
            simple_alnum_match = is_simple_alnum_text_correct(prediction, ground_truth)
            if simple_alnum_match is not None:
                rewards.append(float(simple_alnum_match))
                continue
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            yes_no_match = is_yes_no_correct(content, sol)
            if yes_no_match is not None:
                rewards.append(float(yes_no_match))
                continue
            colon_match = is_colon_value_correct(content, sol)
            if colon_match is not None:
                rewards.append(float(colon_match))
                continue
            choice_match = is_multiple_choice_correct(content, sol)
            if choice_match is not None:
                rewards.append(float(choice_match))
                continue
            simple_match = is_simple_text_correct(content, sol)
            if simple_match is not None:
                rewards.append(float(simple_match))
                continue
            simple_alnum_match = is_simple_alnum_text_correct(content, sol)
            if simple_alnum_match is not None:
                rewards.append(float(simple_alnum_match))
                continue
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            content_to_parse = content_match.group(1).strip() if content_match else content
            has_answer_tag = content_match is not None

            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            sol_to_parse = sol_match.group(1).strip() if sol_match else sol

            gold_parsed = parse(sol_to_parse, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                if has_answer_tag:
                    answer_parsed = parse(content_to_parse, extraction_mode='first_match')
                else:
                    answer_parsed = parse(
                        content_to_parse,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed=True,
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode='first_match',
                    )
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>.*?<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, soft_max_length, soft_cache_length):
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
}
