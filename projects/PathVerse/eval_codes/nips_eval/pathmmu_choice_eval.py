import argparse
import re
from collections import OrderedDict
from typing import Dict, List, Tuple

from tabulate import tabulate
from mathruler.grader import extract_boxed_content

from utils import load_json, load_jsonl, extract_between_tags


def extract_gt_answer(item: dict) -> str:
    if "answer" in item and item["answer"] is not None:
        return str(item["answer"])
    messages = item.get("messages") or []
    if messages:
        last = messages[-1]
        if last.get("role") in {"assistant", "gpt", "bot"}:
            return str(last.get("content", ""))
    return str(item.get("solution", ""))


def normalize_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def clean_response_text(text: str) -> str:
    text = text.replace("•", " ")
    text = re.sub(r"[*_`]", "", text)
    return text


def extract_question_text(item: dict) -> str:
    messages = item.get("messages") or []
    user_msgs = [
        m.get("content", "")
        for m in messages
        if m.get("role") in {"user", "human", "tool"}
    ]
    if user_msgs:
        return str(user_msgs[-1])
    for key in ("prompt", "question", "text"):
        if key in item:
            return str(item[key])
    return ""


def extract_options(question_text: str) -> Dict[str, str]:
    if not question_text:
        return {}
    text = question_text.replace("<image>", "\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    options: Dict[str, str] = {}

    line_pattern = re.compile(r"^([A-Z])[\.\):]\s*(.+)$")
    for line in lines:
        match = line_pattern.match(line)
        if match:
            label = match.group(1).upper()
            content = match.group(2).strip()
            if content:
                options[label] = content
    if options:
        return options

    text = normalize_space(text)
    inline_pattern = re.compile(r"(?:^|\\s)([A-Z])[\.\):]\\s+")
    matches = list(inline_pattern.finditer(text))
    if not matches:
        return {}
    for i, match in enumerate(matches):
        label = match.group(1).upper()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip(" ,;，。")
        if content:
            options[label] = content
    return options


def find_label_candidates_with_pos(text: str, valid_labels: List[str]) -> List[Tuple[str, int]]:
    patterns = [
        r"^\s*([A-Z])(?:[\.\):]|\b)",
        r"\(([A-Z])\)",
        r"(?:^|[^A-Za-z0-9])([A-Z])\s*(?:[\.\):])",
    ]
    candidates: List[Tuple[str, int]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            label = match.group(1).upper()
            if label in valid_labels:
                candidates.append((label, match.start()))
    return candidates


def match_by_option_text(response: str, options: Dict[str, str]) -> List[str]:
    response_norm = normalize_space(response.lower())
    matches: List[str] = []
    for label, opt_text in options.items():
        opt_norm = normalize_space(str(opt_text).lower())
        if len(opt_norm) < 4:
            continue
        if opt_norm in response_norm:
            matches.append(label)
    return matches


def match_by_option_keywords(response: str, options: Dict[str, str]):
    response_norm = normalize_space(response.lower())
    response_tokens = {
        tok for tok in re.findall(r"[a-zA-Z]+", response_norm) if len(tok) >= 4
    }
    if not response_tokens:
        return None
    scored = []
    for label, opt_text in options.items():
        opt_tokens = [
            tok
            for tok in re.findall(r"[a-zA-Z]+", str(opt_text).lower())
            if len(tok) >= 4
        ]
        if len(opt_tokens) < 2:
            continue
        overlap = sum(1 for tok in opt_tokens if tok in response_tokens)
        ratio = overlap / len(opt_tokens)
        if overlap >= 2:
            scored.append((label, ratio, overlap))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    if best[1] < 0.6:
        return None
    if second and best[1] - second[1] < 0.2:
        return None
    return best[0]


def normalize_label(text: str, options: Dict[str, str]):
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    text = extract_between_tags(text, "<answer>", "</answer>", return_origin=True)
    boxed = extract_boxed_content(text)
    if boxed and boxed != "None":
        text = boxed
    valid_labels = sorted(options.keys())
    cleaned = clean_response_text(text)
    if valid_labels:
        explicit = re.search(
            r"(?:^|\b)(?:final\s+answer|option|answer|ans|答案|选项|选择)\s*[:：]?\s*([A-Z])\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if explicit:
            label = explicit.group(1).upper()
            if label in valid_labels:
                return label
    candidates = find_label_candidates_with_pos(cleaned, valid_labels)
    if candidates:
        early = [label for label, pos in candidates if pos <= 200]
        if early:
            unique = list(dict.fromkeys(early))
            if len(unique) == 1:
                return unique[0]
        unique_all = list(dict.fromkeys(label for label, _ in candidates))
        if len(unique_all) == 1:
            return unique_all[0]
    if options:
        matches = match_by_option_text(cleaned, options)
        unique = list(dict.fromkeys(matches))
        if len(unique) == 1:
            return unique[0]
        keyword_label = match_by_option_keywords(cleaned, options)
        if keyword_label is not None:
            return keyword_label
    return None


def build_gt_map(gt_data: List[dict]) -> Tuple[Dict[int, dict], List[str]]:
    gt_map: Dict[int, dict] = {}
    category_order: List[str] = []
    for idx, item in enumerate(gt_data):
        qid = item.get("question_id", idx)
        gt_map[qid] = item
        category = item.get("category", "Unknown")
        if category not in category_order:
            category_order.append(category)
    return gt_map, category_order


def evaluate(result_file: str, gt_file: str) -> Tuple[str, Dict[str, int]]:
    gt_data = load_json(gt_file)
    pred_data = load_jsonl(result_file)

    gt_map, category_order = build_gt_map(gt_data)

    stats = OrderedDict()
    overall_correct = 0
    overall_total = 0
    missing_gt = 0

    for idx, pred in enumerate(pred_data):
        qid = pred.get("question_id", idx)
        gt_item = gt_map.get(qid)
        if gt_item is None:
            missing_gt += 1
            continue
        category = gt_item.get("category", "Unknown")
        if category not in stats:
            stats[category] = {"correct": 0, "total": 0}
        question_text = extract_question_text(gt_item)
        options = extract_options(question_text)
        gt_label = normalize_label(extract_gt_answer(gt_item), options)
        pred_label = normalize_label(pred.get("model_response", ""), options)
        # print(idx, gt_label, pred_label)

        stats[category]["total"] += 1
        overall_total += 1
        if gt_label is not None and pred_label is not None and gt_label == pred_label:
            stats[category]["correct"] += 1
            overall_correct += 1

    table = []
    for category in category_order:
        if category not in stats:
            continue
        total = stats[category]["total"]
        acc = stats[category]["correct"] / total if total else 0.0
        table.append([category, round(acc * 100, 2), total])

    overall_acc = overall_correct / overall_total if overall_total else 0.0
    table.append(["Overall", round(overall_acc * 100, 2), overall_total])

    report = tabulate(table, headers=["Category", "Acc", "Data Num"], tablefmt="orgtbl")

    extra = {
        "missing_gt": missing_gt,
        "total_pred": len(pred_data),
        "total_gt": len(gt_data),
    }
    return report, extra


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate PathMMU choice accuracy and print a table."
    )
    parser.add_argument("--result_file", required=True, help="Path to model result jsonl")
    parser.add_argument("--gt_file", required=True, help="Path to ground truth json")
    args = parser.parse_args()

    report, extra = evaluate(args.result_file, args.gt_file)
    print(report)
    if extra["missing_gt"]:
        print(
            f"[warn] missing gt: {extra['missing_gt']} | pred: {extra['total_pred']} | gt: {extra['total_gt']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
