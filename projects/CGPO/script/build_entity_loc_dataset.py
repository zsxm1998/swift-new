#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build entity-localization conversations from a dataset with <entity> tags."""

from __future__ import annotations

import argparse
import json
import random
import re
from typing import Dict, List, Tuple


ENTITY_PATTERN = re.compile(
    r"<entity\s+name=(\"|')(?P<name>[^\"']+)\1[^>]*>\s*"
    r"(?P<bbox><bbox_list>.*?</bbox_list>)\s*</entity>",
    re.DOTALL,
)


CN_PROMPTS = [
    "检测图像中的{entity}",
    "请定位图像里的{entity}",
    "找出图像中{entity}的位置",
    "请给出{entity}的检测框",
    "标注图像里的{entity}",
]

EN_PROMPTS = [
    "Locate the {entity} in the image.",
    "Detect the {entity} in the image.",
    "Find the {entity} and provide its bbox.",
    "Please identify the {entity} location.",
    "Provide the bounding box for the {entity}.",
]


def has_cjk(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def extract_entities(messages: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    entities: List[Tuple[str, str]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        for match in ENTITY_PATTERN.finditer(content):
            name = match.group("name").strip()
            bbox = match.group("bbox").strip()
            if name and bbox:
                entities.append((name, bbox))
    return entities


def build_prompt(entity_name: str, rng: random.Random) -> str:
    prompts = CN_PROMPTS if has_cjk(entity_name) else EN_PROMPTS
    template = rng.choice(prompts)
    return template.format(entity=entity_name)


def build_conversation(
    item: Dict[str, object], rng: random.Random
) -> Dict[str, object] | None:
    messages = item.get("messages", [])
    if not isinstance(messages, list):
        return None

    entities = extract_entities(messages)
    if not entities:
        return None

    rng.shuffle(entities)

    new_messages: List[Dict[str, str]] = []
    # for msg in messages:
    #     if msg.get("role") == "system":
    #         new_messages.append({"role": "system", "content": msg.get("content", "")})
    #     else:
    #         break

    for idx, (name, bbox) in enumerate(entities):
        prompt = build_prompt(name, rng)
        if idx == 0:
            prompt = "<image>\n" + prompt
        new_messages.append({"role": "user", "content": prompt})
        new_messages.append({"role": "assistant", "content": bbox})

    return {
        "images": item.get("images", []),
        "messages": new_messages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build entity-localization conversations from dataset JSON."
    )
    parser.add_argument("--input", required=True, help="Path to input JSON dataset.")
    parser.add_argument("--output", required=True, help="Path to output JSON dataset.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of conversation items.")

    new_data: List[Dict[str, object]] = []
    skipped = 0
    for item in data:
        if not isinstance(item, dict):
            skipped += 1
            continue
        new_item = build_conversation(item, rng)
        if new_item is None:
            skipped += 1
            continue
        new_data.append(new_item)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(new_data)} items; skipped {skipped}.")


if __name__ == "__main__":
    main()
