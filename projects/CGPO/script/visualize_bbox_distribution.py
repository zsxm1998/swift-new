#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ENTITY_RE = re.compile(r"<entity\b[^>]*>.*?</entity>", re.DOTALL)
BBOX_LIST_RE = re.compile(r"<bbox_list>(.*?)</bbox_list>", re.DOTALL)
BOX_RE = re.compile(r"<box>(.*?)</box>", re.DOTALL)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def clamp_coord(value: float, low: float = 0.0, high: float = 1000.0) -> float:
    return max(low, min(high, value))


def parse_boxes_from_text(text: str) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    for entity_block in ENTITY_RE.findall(text):
        for bbox_block in BBOX_LIST_RE.findall(entity_block):
            for box_text in BOX_RE.findall(bbox_block):
                nums = [float(x) for x in NUM_RE.findall(box_text)]
                if len(nums) >= 4:
                    x1, y1, x2, y2 = nums[:4]
                    x1 = clamp_coord(x1)
                    y1 = clamp_coord(y1)
                    x2 = clamp_coord(x2)
                    y2 = clamp_coord(y2)
                    boxes.append((x1, y1, x2, y2))
    return boxes


def read_boxes(jsonl_path: str) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("model_response", "")
            if "<entity" not in text or "bbox_list" not in text:
                continue
            boxes.extend(parse_boxes_from_text(text))
    return boxes


def plot_distribution(
    boxes: List[Tuple[float, float, float, float]], out_path: str
) -> None:
    xs1, ys1, xs2, ys2, ws, hs = [], [], [], [], [], []
    for x1, y1, x2, y2 in boxes:
        xs1.append(x1)
        ys1.append(y1)
        xs2.append(x2)
        ys2.append(y2)
        ws.append(max(0.0, x2 - x1))
        hs.append(max(0.0, y2 - y1))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    data = [
        (xs1, "x1"),
        (ys1, "y1"),
        (xs2, "x2"),
        (ys2, "y2"),
        (ws, "width"),
        (hs, "height"),
    ]
    for ax, (vals, label) in zip(axes.ravel(), data):
        ax.hist(vals, bins=50, color="#4c72b0", alpha=0.9)
        ax.set_title(label)
        ax.set_xlim(0, 1000)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def plot_spatial_distribution(
    boxes: List[Tuple[float, float, float, float]],
    out_path: str,
    bins: int = 60,
) -> None:
    if not boxes:
        return
    cx = [(x1 + x2) / 2 for x1, _, x2, _ in boxes]
    cy = [(y1 + y2) / 2 for _, y1, _, y2 in boxes]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.hist2d(cx, cy, bins=bins, range=[[0, 1000], [0, 1000]], cmap="magma")
    ax.set_title("center distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1000)
    ax.set_ylim(1000, 0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def plot_box_overlay(
    boxes: List[Tuple[float, float, float, float]],
    out_path: str,
    max_boxes: int = 3000,
    alpha: float = 0.03,
) -> None:
    if not boxes:
        return
    if len(boxes) > max_boxes:
        random.seed(42)
        boxes = random.sample(boxes, max_boxes)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for x1, y1, x2, y2 in boxes:
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w == 0 or h == 0:
            continue
        rect = Rectangle((x1, y1), w, h, fill=False, edgecolor="#1f77b4", linewidth=0.6, alpha=alpha)
        ax.add_patch(rect)
    ax.set_title("box overlay")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1000)
    ax.set_ylim(1000, 0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract bbox_list from <entity> in model_response and visualize distributions."
    )
    parser.add_argument("jsonl_path", help="Path to input jsonl file.")
    parser.add_argument(
        "--out-name",
        default="bbox_distribution.png",
        help="Output image name (saved under input_dir/vis).",
    )
    parser.add_argument(
        "--out-heatmap-name",
        default="bbox_spatial_heatmap.png",
        help="Spatial heatmap image name (saved under input_dir/vis).",
    )
    parser.add_argument(
        "--out-overlay-name",
        default="bbox_overlay.png",
        help="Box overlay image name (saved under input_dir/vis).",
    )
    args = parser.parse_args()

    jsonl_path = args.jsonl_path
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"Input file not found: {jsonl_path}")

    boxes = read_boxes(jsonl_path)
    if not boxes:
        print("No bbox found. Nothing to visualize.")
        return 0

    out_dir = os.path.join(os.path.dirname(jsonl_path), "vis")
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.splitext(os.path.basename(jsonl_path))[0]
    out_path = os.path.join(out_dir, f"{prefix}_{args.out_name}")
    plot_distribution(boxes, out_path)
    heatmap_path = os.path.join(out_dir, f"{prefix}_{args.out_heatmap_name}")
    plot_spatial_distribution(boxes, heatmap_path)
    overlay_path = os.path.join(out_dir, f"{prefix}_{args.out_overlay_name}")
    plot_box_overlay(boxes, overlay_path)
    print(f"Saved visualization to: {out_path}")
    print(f"Saved spatial heatmap to: {heatmap_path}")
    print(f"Saved box overlay to: {overlay_path}")
    print(f"Total boxes: {len(boxes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
