#!/usr/bin/env python3
import argparse
import io
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image


BOX_RE = re.compile(r"<box>(.*?)</box>", re.DOTALL)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def clamp_coord(value: float, low: float = 0.0, high: float = 1000.0) -> float:
    return max(low, min(high, value))


def parse_boxes_from_text(text: str) -> List[List[float]]:
    if not text:
        return []
    boxes: List[List[float]] = []
    for box_text in BOX_RE.findall(text):
        nums = [float(x) for x in NUM_RE.findall(box_text)]
        if len(nums) >= 4:
            x1, y1, x2, y2 = nums[:4]
            boxes.append(
                [
                    clamp_coord(x1),
                    clamp_coord(y1),
                    clamp_coord(x2),
                    clamp_coord(y2),
                ]
            )
    return boxes


def extract_user_question(messages: Optional[List[dict]]) -> str:
    if not messages:
        return ""
    parts = []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if content:
                parts.append(content)
    return "\n\n".join(parts)


def build_group_key(image_path: str, user_question: str) -> str:
    return f"{image_path}||{user_question}"


def extract_assistant_response(obj: dict) -> str:
    if "model_response" in obj and obj.get("model_response"):
        return obj.get("model_response", "")
    messages = obj.get("messages", [])
    if isinstance(messages, list):
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
    return ""


def find_jsonl_files(root_dir: str) -> List[str]:
    matches = []
    for current, _, files in os.walk(root_dir):
        for name in files:
            if name == "cgpo_entity_det.jsonl":
                matches.append(os.path.join(current, name))
    matches.sort()
    return matches


def get_image_path(obj: dict) -> str:
    images = obj.get("images", [])
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict):
            return first.get("path", "") or ""
        if isinstance(first, str):
            return first
    return ""


def normalize_step(value: object) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def scan_steps(jsonl_path: str) -> List[str]:
    steps = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = normalize_step(obj.get("step"))
            if step is not None:
                steps.add(step)
    return sorted(steps, key=lambda x: (len(x), x))


def load_step_data(jsonl_path: str, step: str) -> List[dict]:
    groups: Dict[str, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            obj_step = normalize_step(obj.get("step"))
            if obj_step != step:
                continue
            prompt_id = str(obj.get("prompt_id", ""))
            messages = obj.get("messages", [])
            image_path = get_image_path(obj)
            user_question = extract_user_question(messages)
            group_key = build_group_key(image_path, user_question)
            if not image_path and not user_question:
                continue
            solution = obj.get("solution", "") or ""
            group = groups.setdefault(
                group_key,
                {
                    "group_key": group_key,
                    "prompt_ids": [],
                    "user_question": "",
                    "solution": "",
                    "image_path": "",
                    "responses": [],
                },
            )
            if prompt_id and prompt_id not in group["prompt_ids"]:
                group["prompt_ids"].append(prompt_id)
            if user_question and not group["user_question"]:
                group["user_question"] = user_question
            if solution and not group["solution"]:
                group["solution"] = solution
            if image_path and not group["image_path"]:
                group["image_path"] = image_path

            entities = []
            for ent in obj.get("entities", []) or []:
                name = ent.get("name", "")
                gt_boxes = parse_boxes_from_text(ent.get("gt_bbox_text", ""))
                pred_boxes = parse_boxes_from_text(ent.get("pred_bbox_text", ""))
                entities.append(
                    {
                        "name": name,
                        "gt_boxes": gt_boxes,
                        "pred_boxes": pred_boxes,
                    }
                )
            response = {
                "model_response": extract_assistant_response(obj),
                "entities": entities,
            }
            group["responses"].append(response)
    return list(groups.values())


def build_prompt_index(jsonl_path: str, step: str) -> List[str]:
    group_keys: List[str] = []
    seen = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            obj_step = normalize_step(obj.get("step"))
            if obj_step != step:
                continue
            image_path = get_image_path(obj)
            user_question = extract_user_question(obj.get("messages", []))
            group_key = build_group_key(image_path, user_question)
            if not image_path and not user_question:
                continue
            if group_key in seen:
                continue
            seen.add(group_key)
            group_keys.append(group_key)
    return group_keys


def load_page_data(jsonl_path: str, step: str, target_ids: List[str]) -> List[dict]:
    target_set = set(target_ids)
    groups: Dict[str, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            obj_step = normalize_step(obj.get("step"))
            if obj_step != step:
                continue
            prompt_id = str(obj.get("prompt_id", ""))
            messages = obj.get("messages", [])
            user_question = extract_user_question(messages)
            image_path = get_image_path(obj)
            group_key = build_group_key(image_path, user_question)
            if not image_path and not user_question:
                continue
            if group_key not in target_set:
                continue
            solution = obj.get("solution", "") or ""
            group = groups.setdefault(
                group_key,
                {
                    "group_key": group_key,
                    "prompt_ids": [],
                    "user_question": "",
                    "solution": "",
                    "image_path": "",
                    "responses": [],
                },
            )
            if prompt_id and prompt_id not in group["prompt_ids"]:
                group["prompt_ids"].append(prompt_id)
            if user_question and not group["user_question"]:
                group["user_question"] = user_question
            if solution and not group["solution"]:
                group["solution"] = solution
            if image_path and not group["image_path"]:
                group["image_path"] = image_path

            entities = []
            for ent in obj.get("entities", []) or []:
                name = ent.get("name", "")
                gt_boxes = parse_boxes_from_text(ent.get("gt_bbox_text", ""))
                pred_boxes = parse_boxes_from_text(ent.get("pred_bbox_text", ""))
                entities.append(
                    {
                        "name": name,
                        "gt_boxes": gt_boxes,
                        "pred_boxes": pred_boxes,
                    }
                )
            response = {
                "model_response": extract_assistant_response(obj),
                "entities": entities,
            }
            group["responses"].append(response)
    return [groups[group_key] for group_key in target_ids if group_key in groups]


def resize_image(path: str, max_size: int) -> Tuple[bytes, str]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        if max(width, height) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue(), "image/jpeg"


def build_app(file_index: List[str], max_image_size: int) -> FastAPI:
    app = FastAPI()
    app.state.file_index = file_index
    app.state.step_cache = {}
    app.state.prompt_index_cache = {}
    app.state.max_image_size = max_image_size

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>CGPO Entity Det Viewer</title>
    <style>
      :root {
        --bg: #f6f2ed;
        --ink: #1f1a17;
        --muted: #6b5d52;
        --accent: #c77d2b;
        --accent-2: #297373;
        --card: #fffdf9;
        --border: #e3d6c8;
      }
      body {
        margin: 0;
        font-family: "IBM Plex Sans", "Work Sans", "Helvetica Neue", Arial, sans-serif;
        color: var(--ink);
        background: radial-gradient(circle at top, #fff4e6 0%, var(--bg) 45%, #efe8e1 100%);
      }
      header {
        padding: 28px 32px 18px 32px;
        border-bottom: 1px solid var(--border);
        background: rgba(255, 253, 249, 0.85);
        backdrop-filter: blur(6px);
        position: sticky;
        top: 0;
        z-index: 10;
      }
      h1 {
        margin: 0 0 6px 0;
        font-size: 24px;
        letter-spacing: 0.5px;
      }
      .sub {
        color: var(--muted);
        font-size: 14px;
      }
      main {
        padding: 24px 32px 48px 32px;
      }
      .controls {
        display: grid;
        grid-template-columns: minmax(220px, 1fr) minmax(140px, 220px) minmax(140px, 180px) auto;
        gap: 12px;
        align-items: center;
        margin-bottom: 12px;
      }
      .pagination {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-bottom: 18px;
      }
      select, button, input[type="number"] {
        padding: 10px 12px;
        font-size: 14px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: #fff;
        color: var(--ink);
      }
      button {
        background: var(--accent);
        color: #fff;
        border: none;
        cursor: pointer;
      }
      button:disabled {
        background: #cdbcae;
        cursor: not-allowed;
      }
      .status {
        margin: 8px 0 20px 0;
        color: var(--muted);
      }
      .group-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(63, 50, 39, 0.08);
      }
      .group-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 12px;
        margin-bottom: 10px;
      }
      .pill {
        display: inline-block;
        padding: 4px 10px;
        font-size: 12px;
        border-radius: 999px;
        background: #efe3d6;
        color: var(--muted);
      }
      .section {
        margin: 12px 0;
      }
      .label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--muted);
        margin-bottom: 6px;
      }
      pre {
        background: #f8f3ed;
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #eadfd3;
        white-space: pre-wrap;
        word-break: break-word;
        margin: 0;
        font-size: 13px;
        line-height: 1.5;
      }
      .table-wrap {
        overflow-x: auto;
      }
      table.group-table {
        width: 100%;
        border-collapse: collapse;
      }
      table.group-table td, table.group-table th {
        border: 1px solid #eadfd3;
        vertical-align: top;
        padding: 10px;
        background: #fffdf9;
      }
      table.group-table th {
        text-align: left;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--muted);
        background: #f8f3ed;
      }
      .left-cell {
        width: 34%;
        min-width: 260px;
      }
      .response-cell pre {
        background: #fdf7f0;
      }
      .img-cell {
        width: 380px;
        min-width: 300px;
      }
      .entity-block {
        margin-bottom: 12px;
      }
      .entity-title {
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 6px;
      }
      .image-wrap {
        background: #f3ece5;
        border-radius: 16px;
        padding: 10px;
        border: 1px solid #eadfd3;
        max-width: 360px;
      }
      canvas {
        width: 100%;
        height: auto;
        display: block;
        border-radius: 12px;
      }
      .legend {
        display: flex;
        gap: 12px;
        font-size: 12px;
        color: var(--muted);
        margin-top: 8px;
      }
      .no-entities {
        font-size: 12px;
        color: var(--muted);
        padding: 8px 0;
      }
      .legend span::before {
        content: "";
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 3px;
        margin-right: 6px;
      }
      .legend .gt::before { background: #3da35d; }
      .legend .pred::before { background: #d94f4f; }
      @media (max-width: 900px) {
        .controls {
          grid-template-columns: 1fr;
        }
        .pagination {
          flex-wrap: wrap;
        }
        header, main {
          padding: 18px;
        }
        .img-cell {
          width: auto;
        }
        .image-wrap {
          max-width: 100%;
        }
      }
    </style>
  </head>
  <body>
    <header>
      <h1>CGPO Entity Detection Viewer</h1>
      <div class="sub">Select a result file and step to visualize grouped prompts and bounding boxes.</div>
    </header>
    <main>
      <div class="controls">
        <select id="fileSelect"></select>
        <select id="stepSelect"></select>
        <select id="pageSizeSelect">
          <option value="3">3 prompts / page</option>
          <option value="5" selected>5 prompts / page</option>
          <option value="10">10 prompts / page</option>
          <option value="20">20 prompts / page</option>
        </select>
        <button id="loadBtn" disabled>Load</button>
      </div>
      <div class="pagination">
        <button id="prevBtnTop" disabled>Prev</button>
        <input id="pageInputTop" type="number" min="1" value="1"/>
        <span id="pageInfoTop">/ 1</span>
        <button id="nextBtnTop" disabled>Next</button>
      </div>
      <div class="status" id="status">Loading file list...</div>
      <div id="content"></div>
      <div class="pagination">
        <button id="prevBtnBottom" disabled>Prev</button>
        <input id="pageInputBottom" type="number" min="1" value="1"/>
        <span id="pageInfoBottom">/ 1</span>
        <button id="nextBtnBottom" disabled>Next</button>
      </div>
    </main>
    <script>
      const fileSelect = document.getElementById("fileSelect");
      const stepSelect = document.getElementById("stepSelect");
      const pageSizeSelect = document.getElementById("pageSizeSelect");
      const loadBtn = document.getElementById("loadBtn");
      const prevBtnTop = document.getElementById("prevBtnTop");
      const nextBtnTop = document.getElementById("nextBtnTop");
      const pageInputTop = document.getElementById("pageInputTop");
      const pageInfoTop = document.getElementById("pageInfoTop");
      const prevBtnBottom = document.getElementById("prevBtnBottom");
      const nextBtnBottom = document.getElementById("nextBtnBottom");
      const pageInputBottom = document.getElementById("pageInputBottom");
      const pageInfoBottom = document.getElementById("pageInfoBottom");
      const statusEl = document.getElementById("status");
      const contentEl = document.getElementById("content");
      let currentPage = 1;
      let totalPages = 1;
      let lastLoaded = { fileId: null, step: null };

      function escapeHtml(value) {
        if (value === null || value === undefined) return "";
        return String(value)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }

      function setStatus(text) {
        statusEl.textContent = text;
      }

      function clearSelect(select) {
        while (select.firstChild) select.removeChild(select.firstChild);
      }

      async function fetchFiles() {
        const res = await fetch("/api/files");
        const data = await res.json();
        clearSelect(fileSelect);
        data.files.forEach((file) => {
          const opt = document.createElement("option");
          opt.value = file.id;
          opt.textContent = file.label;
          fileSelect.appendChild(opt);
        });
        if (data.files.length === 0) {
          setStatus("No cgpo_entity_det.jsonl found.");
          loadBtn.disabled = true;
          updatePagination(0, 1, Number(pageSizeSelect.value));
          return;
        }
        setStatus("Select a file to load available steps.");
        await fetchSteps();
      }

      async function fetchSteps() {
        const fileId = fileSelect.value;
        clearSelect(stepSelect);
        setStatus("Scanning steps, please wait...");
        const res = await fetch(`/api/steps?file_id=${encodeURIComponent(fileId)}`);
        const data = await res.json();
        data.steps.forEach((step) => {
          const opt = document.createElement("option");
          opt.value = step;
          opt.textContent = step;
          stepSelect.appendChild(opt);
        });
        if (data.steps.length === 0) {
          setStatus("No steps found in this file.");
          loadBtn.disabled = true;
          updatePagination(0, 1, Number(pageSizeSelect.value));
          return;
        }
        loadBtn.disabled = false;
        currentPage = 1;
        lastLoaded = { fileId: null, step: null };
        updatePagination(0, 1, Number(pageSizeSelect.value));
        setStatus("Ready to load data.");
      }

      function renderGroups(groups) {
        contentEl.innerHTML = "";
        if (groups.length === 0) {
          contentEl.innerHTML = "<div class='group-card'>No data for this step.</div>";
          return;
        }
        groups.forEach((group, idx) => {
          const card = document.createElement("div");
          card.className = "group-card";
          const responses = group.responses || [];
          const rowCount = Math.max(1, responses.length);
          const tableWrap = document.createElement("div");
          tableWrap.className = "table-wrap";
          const table = document.createElement("table");
          table.className = "group-table";
          table.innerHTML = `
            <thead>
              <tr>
                <th>Prompt</th>
                <th>Response Text</th>
                <th>Detection Overlay</th>
              </tr>
            </thead>
            <tbody></tbody>
          `;
          const tbody = table.querySelector("tbody");
          for (let rIdx = 0; rIdx < rowCount; rIdx += 1) {
            const resp = responses[rIdx] || { model_response: "", entities: [] };
            const row = document.createElement("tr");
            if (rIdx === 0) {
              const leftCell = document.createElement("td");
              leftCell.className = "left-cell";
              leftCell.rowSpan = rowCount;
              leftCell.innerHTML = `
                <div class="group-header">
                  <div><strong>Prompt Group</strong></div>
                  <span class="pill">prompt_id: ${(group.prompt_ids || []).join(", ")}</span>
                </div>
                <div class="section">
                  <div class="label">User Question</div>
                  <pre>${escapeHtml(group.user_question || "")}</pre>
                </div>
                <div class="section">
                  <div class="label">Solution</div>
                  <pre>${escapeHtml(group.solution || "")}</pre>
                </div>
              `;
              row.appendChild(leftCell);
            }
            const responseId = `canvas_${idx}_${rIdx}`;
            const respCell = document.createElement("td");
            respCell.className = "response-cell";
            respCell.innerHTML = `
              <div class="pill">#${rIdx + 1}</div>
              <pre>${escapeHtml(resp.model_response || "")}</pre>
            `;
            const imgCell = document.createElement("td");
            imgCell.className = "img-cell";
            const entities = resp.entities || [];
            if (entities.length === 0) {
              imgCell.innerHTML = `<div class="no-entities">No entities.</div>`;
            } else {
              const blocks = entities.map((ent, eIdx) => {
                const entityId = `${responseId}_${eIdx}`;
                return `
                  <div class="entity-block">
                    <div class="entity-title">${escapeHtml(ent.name || "entity")}</div>
                    <div class="image-wrap">
                      <canvas id="${entityId}"></canvas>
                      <div class="legend">
                        <span class="gt">GT</span>
                        <span class="pred">Pred</span>
                      </div>
                    </div>
                  </div>
                `;
              }).join("");
              imgCell.innerHTML = blocks;
            }
            row.appendChild(respCell);
            row.appendChild(imgCell);
            tbody.appendChild(row);
            if (entities.length > 0) {
              entities.forEach((ent, eIdx) => {
                const entityId = `${responseId}_${eIdx}`;
                setTimeout(() => drawCanvas(entityId, group.image_path, ent), 0);
              });
            }
          }
          tableWrap.appendChild(table);
          card.appendChild(tableWrap);
          contentEl.appendChild(card);
        });
      }

      function drawCanvas(canvasId, imagePath, entity) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !imagePath) {
          return;
        }
        const ctx = canvas.getContext("2d");
        const img = new Image();
        img.onload = () => {
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          ctx.drawImage(img, 0, 0);
          const scaleX = img.naturalWidth / 1000.0;
          const scaleY = img.naturalHeight / 1000.0;
          const drawBoxes = (boxes, color) => {
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            boxes.forEach((box) => {
              const [x1, y1, x2, y2] = box;
              const w = Math.max(0, (x2 - x1) * scaleX);
              const h = Math.max(0, (y2 - y1) * scaleY);
              ctx.strokeRect(x1 * scaleX, y1 * scaleY, w, h);
            });
          };
          drawBoxes(entity.gt_boxes || [], "#3da35d");
          drawBoxes(entity.pred_boxes || [], "#d94f4f");
        };
        img.src = `/api/image?path=${encodeURIComponent(imagePath)}`;
      }

      function updatePagination(total, page, pageSize) {
        totalPages = Math.max(1, Math.ceil(total / pageSize));
        currentPage = Math.min(page, totalPages);
        pageInputTop.value = currentPage;
        pageInputBottom.value = currentPage;
        const infoText = `/ ${totalPages} (${total} prompts)`;
        pageInfoTop.textContent = infoText;
        pageInfoBottom.textContent = infoText;
        const isFirst = currentPage <= 1;
        const isLast = currentPage >= totalPages;
        prevBtnTop.disabled = isFirst;
        prevBtnBottom.disabled = isFirst;
        nextBtnTop.disabled = isLast;
        nextBtnBottom.disabled = isLast;
      }

      async function loadPage(page, scrollToTop = true) {
        const fileId = fileSelect.value;
        const step = stepSelect.value;
        const pageSize = Number(pageSizeSelect.value);
        if (!fileId || !step) return;
        setStatus("Loading data, this may take a while for large files...");
        loadBtn.disabled = true;
        const url = `/api/data?file_id=${encodeURIComponent(fileId)}&step=${encodeURIComponent(step)}&page=${page}&page_size=${pageSize}`;
        const res = await fetch(url);
        const data = await res.json();
        renderGroups(data.groups || []);
        updatePagination(data.total || 0, data.page || 1, data.page_size || pageSize);
        setStatus(`Loaded page ${currentPage} / ${totalPages} for step ${step}.`);
        loadBtn.disabled = false;
        lastLoaded = { fileId, step };
        if (scrollToTop) {
          window.scrollTo({ top: 0, behavior: "smooth" });
        }
      }

      loadBtn.addEventListener("click", async () => {
        currentPage = 1;
        await loadPage(currentPage, true);
      });

      prevBtnTop.addEventListener("click", async () => {
        if (currentPage > 1) {
          await loadPage(currentPage - 1, true);
        }
      });

      nextBtnTop.addEventListener("click", async () => {
        if (currentPage < totalPages) {
          await loadPage(currentPage + 1, true);
        }
      });

      pageInputTop.addEventListener("change", async () => {
        const val = Number(pageInputTop.value);
        if (!Number.isFinite(val)) return;
        const page = Math.min(Math.max(1, val), totalPages);
        await loadPage(page, true);
      });

      prevBtnBottom.addEventListener("click", async () => {
        if (currentPage > 1) {
          await loadPage(currentPage - 1, true);
        }
      });

      nextBtnBottom.addEventListener("click", async () => {
        if (currentPage < totalPages) {
          await loadPage(currentPage + 1, true);
        }
      });

      pageInputBottom.addEventListener("change", async () => {
        const val = Number(pageInputBottom.value);
        if (!Number.isFinite(val)) return;
        const page = Math.min(Math.max(1, val), totalPages);
        await loadPage(page, true);
      });

      pageSizeSelect.addEventListener("change", async () => {
        if (lastLoaded.fileId !== null) {
          currentPage = 1;
          await loadPage(currentPage, true);
        }
      });

      fileSelect.addEventListener("change", async () => {
        loadBtn.disabled = true;
        await fetchSteps();
      });

      stepSelect.addEventListener("change", async () => {
        currentPage = 1;
        lastLoaded = { fileId: null, step: null };
        updatePagination(0, 1, Number(pageSizeSelect.value));
        setStatus("Ready to load data.");
      });

      fetchFiles().catch((err) => {
        setStatus(`Failed to load: ${err}`);
      });
    </script>
  </body>
</html>
"""
        return HTMLResponse(html)

    @app.get("/api/files", response_class=JSONResponse)
    def api_files() -> JSONResponse:
        files = []
        for idx, path in enumerate(app.state.file_index):
            label = os.path.relpath(path, start=app.state.root_dir)
            files.append({"id": idx, "path": path, "label": label})
        return JSONResponse({"files": files})

    @app.get("/api/steps", response_class=JSONResponse)
    def api_steps(file_id: int = Query(...)) -> JSONResponse:
        try:
            jsonl_path = app.state.file_index[file_id]
        except IndexError:
            raise HTTPException(status_code=404, detail="Invalid file_id.")
        mtime = os.path.getmtime(jsonl_path)
        cache = app.state.step_cache.get(jsonl_path)
        if cache and cache["mtime"] == mtime:
            return JSONResponse({"steps": cache["steps"]})
        steps = scan_steps(jsonl_path)
        app.state.step_cache[jsonl_path] = {"mtime": mtime, "steps": steps}
        return JSONResponse({"steps": steps})

    @app.get("/api/data", response_class=JSONResponse)
    def api_data(
        file_id: int = Query(...),
        step: str = Query(...),
        page: int = Query(1, ge=1),
        page_size: int = Query(5, ge=1, le=100),
    ) -> JSONResponse:
        try:
            jsonl_path = app.state.file_index[file_id]
        except IndexError:
            raise HTTPException(status_code=404, detail="Invalid file_id.")
        cache_key = (jsonl_path, step)
        mtime = os.path.getmtime(jsonl_path)
        cache = app.state.prompt_index_cache.get(cache_key)
        if cache and cache["mtime"] == mtime:
            prompt_ids = cache["prompt_ids"]
        else:
            start = time.time()
            prompt_ids = build_prompt_index(jsonl_path, step)
            app.state.prompt_index_cache[cache_key] = {
                "mtime": mtime,
                "prompt_ids": prompt_ids,
                "loaded_at": start,
            }
        total = len(prompt_ids)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_ids = prompt_ids[start_idx:end_idx]
        page_groups = load_page_data(jsonl_path, step, page_ids)
        return JSONResponse(
            {
                "groups": page_groups,
                "page": page,
                "page_size": page_size,
                "total": total,
            }
        )

    @app.get("/api/image")
    def api_image(path: str = Query(...)) -> Response:
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail="Image not found.")
        try:
            data, content_type = resize_image(abs_path, app.state.max_image_size)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load image: {exc}")
        return Response(content=data, media_type=content_type)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Web viewer for cgpo_entity_det.jsonl visualization."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7000,
        help="Port to run the server on.",
    )
    parser.add_argument(
        "--root",
        default="outputs",
        help="Root directory to scan for cgpo_entity_det.jsonl files.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=600,
        help="Max width/height of images served to the browser.",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    file_index = find_jsonl_files(root_dir)
    app = build_app(file_index, args.max_image_size)
    app.state.root_dir = root_dir

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
