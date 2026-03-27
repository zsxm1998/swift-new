#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add solution and task fields to RefPath-style conversation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def find_last_assistant_content(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str):
                return content
            raise ValueError("Assistant content is not a string")
    raise ValueError("No assistant message found")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the last assistant message content into solution and add task"
        )
    )
    parser.add_argument("--input_json", required=True, help="Input JSON file")
    parser.add_argument(
        "--output_json",
        default=None,
        help="Output JSON file (default: <input>_with_solution.json)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input JSON file in-place",
    )
    parser.add_argument("--task", required=True, help="Task name to set")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if args.inplace and args.output_json is not None:
        raise ValueError("Cannot use --inplace together with --output_json")
    if args.inplace:
        output_path = input_path
    elif args.output_json is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_with_solution{input_path.suffix}"
        )
    else:
        output_path = Path(args.output_json)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array")

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each item must be a JSON object")
        messages = item.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Item missing messages list")
        item["solution"] = find_last_assistant_content(messages)
        item["task"] = args.task

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} items to {output_path}")


if __name__ == "__main__":
    main()
