#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build RefPath referring-detection dataset with normalized 0-1000 bboxes."""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path
from typing import Dict, List

from huggingface_hub import hf_hub_download


CN_PROMPTS = [
    "检测图像中的{expr}",
    "检测图像中描述为“{expr}”的区域",
    "请定位：{expr}",
    "在图像中找出{expr}",
    "请给出“{expr}”的检测框",
]

EN_PROMPTS = [
    "Detect the {expr} in the image",
    "Locate the region described as: {expr}",
    "Find the area matching: {expr}",
    "Detect the region: {expr}",
    "Provide the bounding box for: {expr}",
]


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def build_prompt(expr: str, rng: random.Random, force_first: bool = False) -> str:
    prompts = CN_PROMPTS if has_cjk(expr) else EN_PROMPTS
    template = prompts[0] if force_first else rng.choice(prompts)
    return template.format(expr=expr)


def normalize_expression(expr: str) -> str:
    expr = expr.strip()
    if expr:
        expr = expr[0].lower() + expr[1:]
    expr = expr.rstrip(" .。!?！？…")
    return expr


def clamp_int(value: float, low: int = 0, high: int = 1000) -> int:
    return max(low, min(high, int(round(value))))


def normalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")
    x1, y1, x2, y2 = bbox
    return [
        clamp_int(x1 / width * 1000),
        clamp_int(y1 / height * 1000),
        clamp_int(x2 / width * 1000),
        clamp_int(y2 / height * 1000),
    ]


def safe_extract(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in zip: {member.filename}")
            out_path = target_dir / member_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(out_path, "wb") as dst:
                dst.write(src.read())


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare RefPath referring-detection dataset."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for images and conversations JSON.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "testA", "testB"],
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional output JSON path. Defaults to output_dir/refpath_<split>_convs.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for prompt selection and expression order.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples for debugging.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    images_root = output_dir
    images_dir = output_dir / "refpath_image"
    images_dir.mkdir(parents=True, exist_ok=True)

    json_name = args.output_json or str(
        output_dir / f"refpath_{args.split}_convs.json"
    )

    jsonl_name = f"{args.split}.jsonl"
    jsonl_path = hf_hub_download(
        repo_id="fengluo/RefPath", repo_type="dataset", filename=jsonl_name
    )

    zip_path = hf_hub_download(
        repo_id="fengluo/RefPath", repo_type="dataset", filename="refpath_image.zip"
    )
    if not images_dir.exists() or not any(images_dir.iterdir()):
        safe_extract(Path(zip_path), images_root)

    records = load_jsonl(Path(jsonl_path))
    if args.limit is not None:
        records = records[: args.limit]

    conversations: List[Dict[str, object]] = []
    force_first_prompt = args.split in {"testA", "testB"}
    for record in records:
        bbox = record["bbox"]
        width = int(record["width"])
        height = int(record["height"])
        expressions = record.get("expression", [])
        image_rel = record["image"]

        if not expressions:
            continue

        norm_bbox = normalize_bbox(bbox, width, height)
        bbox_text = (
            f"<bbox_list><box>{norm_bbox[0]}, {norm_bbox[1]}, "
            f"{norm_bbox[2]}, {norm_bbox[3]}</box></bbox_list>"
        )

        image_path = images_root / image_rel
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        expressions = [normalize_expression(expr) for expr in expressions]
        expressions = [expr for expr in expressions if expr]
        rng.shuffle(expressions)

        messages: List[Dict[str, str]] = []
        for idx, expr in enumerate(expressions):
            prompt = build_prompt(expr, rng, force_first=force_first_prompt)
            if idx == 0:
                prompt = "<image>\n" + prompt
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": bbox_text})

        conversations.append(
            {"images": [str(image_path)], "messages": messages}
        )

    with open(json_name, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(conversations)} items to {json_name}")


if __name__ == "__main__":
    main()
