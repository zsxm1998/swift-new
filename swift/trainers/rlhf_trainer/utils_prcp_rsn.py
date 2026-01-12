import re
import hashlib
from copy import deepcopy
from functools import partial
from typing import Dict, List, Any, TYPE_CHECKING

import numpy as np
from PIL import Image

from .rollout_mixin import DataType
from ...llm.template.vision_utils import load_image
if TYPE_CHECKING:
    from .grpo_trainer import GRPOTrainer


# ---------------- 图像corruption入口函数和相关工具 ----------------
def _stable_prompt_seed(prompt_id: str, salt: str = "corrupt_prompt_v1") -> int:
    s = f"{salt}:{prompt_id}".encode("utf-8")
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def _bytes_sig(b: bytes):
    if len(b) <= 8192:
        data = b
    else:
        data = b[:4096] + b[-4096:]
    return hashlib.blake2b(data, digest_size=16).digest()


def _image_signature(imgs):
    sig = []
    for img in imgs:
        if isinstance(img, dict):
            img = (img.get('bytes') or img.get('path')) if 'bytes' in img else img['path']
        if isinstance(img, (bytes, bytearray)):
            sig.append(("bytes", _bytes_sig(img)))
        else:
            sig.append(("path", str(img)))
    return tuple(sig)


# ---------------- 入口函数 ----------------
def get_corrupted_images_for_input(trainer: 'GRPOTrainer', inputs: DataType) -> DataType:
    corrupt_image: str = trainer.corrupt_image
    corrupt_image_kwargs: Dict[str, Any] = trainer.corrupt_image_kwargs
    corrupt_image_position: str = trainer.corrupt_image_position

    # 根据选择的corruption方法做一些预处理
    if corrupt_image == 'no_image':
        return inputs
    elif corrupt_image == 'random_patch':
        corrupt_func = partial(random_patch_blackening, patch_size=trainer.model.config.vision_config.patch_size, **corrupt_image_kwargs)
    elif corrupt_image == 'cgpo_v1':
        assert corrupt_image_position == 'completion', "cgpo_v1 only supports corrupt_image_position='completion'"
        corrupt_func = partial(cgpo_v1, **corrupt_image_kwargs)
    elif corrupt_image == 'cgpo_v2':
        assert corrupt_image_position == 'completion', "cgpo_v2 only supports corrupt_image_position='completion'"
        corrupt_func = partial(cgpo_v2, **corrupt_image_kwargs)
    else:
        raise NotImplementedError(f"Corrupt image method '{corrupt_image}' is not implemented.")
    
    # 回答有关的图像corruption操作
    if corrupt_image in ['cgpo_v1', 'cgpo_v2']:
        for inp in inputs:
            if 'images' in inp:
                inp['corrupted_images'] = corrupt_func(inp['images'], inp['messages'])
        return inputs

    # 回答无关的图像corruption操作
    if corrupt_image_position == 'completion':
        for inp in inputs:
            if 'images' in inp:
                corrupted_images = []
                for img in inp['images']:
                    if isinstance(img, dict):
                        img = (img.get('bytes') or img.get('path')) if 'bytes' in img else img['path']
                    img = load_image(img)
                    corrupted_images.append(corrupt_func(img))
                inp['corrupted_images'] = corrupted_images

    elif corrupt_image_position == 'prompt':
        prompt_cache: Dict[Any, Any] = {}
        for inp in inputs:
            if 'images' not in inp:
                continue
            if 'prompt_id' not in inp:
                raise KeyError("corrupt_image_position='prompt' requires each input to have 'prompt_id'.")

            pid = inp['prompt_id'] + f'_step:{trainer._step}'
            img_sig = _image_signature(inp["images"])
            if (pid, img_sig) in prompt_cache:
                inp['corrupted_images'] = deepcopy(prompt_cache[(pid, img_sig)])
                continue

            seed_base = _stable_prompt_seed(pid)
            corrupted_images = []

            for img_idx, img in enumerate(inp['images']):
                if isinstance(img, dict):
                    img = (img.get('bytes') or img.get('path')) if 'bytes' in img else img['path']
                img = load_image(img)
                corrupted_images.append(corrupt_func(img, seed=seed_base+img_idx))

            prompt_cache[(pid, img_sig)] = corrupted_images
            inp['corrupted_images'] = corrupted_images
    
    else:
        raise ValueError(f"'corrupt_image_position' can only be 'prompt' or 'completion', got '{corrupt_image_position}'.")
    
    return inputs


# ---------------- 对话内容无关的图像corruption方法 ----------------
def random_patch_blackening(pil_img, patch_size=14, black_prob=0.6, seed=None):
    """Randomly blacken square patches in a PIL image (deterministic if seed provided)."""
    rng = np.random.default_rng(seed)

    img = np.array(pil_img)
    h, w = img.shape[:2]

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if rng.random() < black_prob:
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                img[y:y_end, x:x_end, ...] = 0

    return Image.fromarray(img)


# ---------------- 对话内容相关的图像corruption方法和相应工具 ----------------
# 正则模式统一定义
ENT_PAT = re.compile(r'<entity\s+name="(?P<name>[^"]+)"\s+id="(?P<id>\d+)">\s*(?P<body>.*?)\s*</entity>', re.DOTALL)
BBOX_LIST_PAT = re.compile(r'<bbox_list>(?P<b>.*?)</bbox_list>', re.DOTALL)
BOX_PAT = re.compile(r'<box>\s*([^<]+?)\s*</box>')
BOX_COORD_PAT = re.compile(r'\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*')


def parse_all_bboxes_from_response(response: str) -> List[List[List[int]]]:
    """从assistant response中按entity id顺序提取三层bbox列表：entity -> bbox_list -> [x1,y1,x2,y2]"""
    # 提取 entity（按 id 排序）
    entities = list(ENT_PAT.finditer(response))
    entities.sort(key=lambda m: int(m.group('id')))

    all_bboxes: List[List[List[int]]] = []  # 三层列表：entity -> bbox_list -> [x1, y1, x2, y2]
    for m in entities:
        body = m.group('body')
        bbox_m = BBOX_LIST_PAT.search(body)
        if not bbox_m:
            all_bboxes.append([])
            continue
        boxes_txt = BOX_PAT.findall(bbox_m.group('b'))
        ent_boxes: List[List[int]] = []
        for t in boxes_txt:
            mm = BOX_COORD_PAT.fullmatch(t)
            if not mm: continue
            ent_boxes.append([int(mm.group(1)), int(mm.group(2)), int(mm.group(3)), int(mm.group(4))])
        all_bboxes.append(ent_boxes)
    return all_bboxes


def to_abs_bbox(x1, y1, x2, y2, W, H):
    """相对坐标(0-1000) -> 像素坐标，并做边界裁剪"""
    ax1 = int(round(x1 / 1000.0 * W))
    ay1 = int(round(y1 / 1000.0 * H))
    ax2 = int(round(x2 / 1000.0 * W))
    ay2 = int(round(y2 / 1000.0 * H))
    ax1 = max(0, min(W, ax1)); ax2 = max(0, min(W, ax2))
    ay1 = max(0, min(H, ay1)); ay2 = max(0, min(H, ay2))
    return ax1, ay1, ax2, ay2


def get_fill_val(arr: np.ndarray, fill_type: str, all_bboxes: List):
    if fill_type == 'black':
        fill_val = 0
    elif fill_type == 'mean':
        fill_val = arr.mean(axis=(0, 1)).astype(arr.dtype)
    elif fill_type == 'keep_mean':
        mask = np.zeros(arr.shape[:2], dtype=bool)
        for ent_boxes in all_bboxes:
            for x1, y1, x2, y2 in ent_boxes:
                ax1, ay1, ax2, ay2 = to_abs_bbox(x1, y1, x2, y2, arr.shape[1], arr.shape[0])
                if ax2 <= ax1 or ay2 <= ay1:
                    continue
                mask[ay1:ay2, ax1:ax2] = True
        if np.any(mask):
            fill_val = np.round(arr[mask].mean(axis=0)).astype(arr.dtype) # 确定为Image类型所以arr一定是整数
        else:
            fill_val = 0
    elif fill_type == 'local_mean':
        fill_val = 'local_mean'  # 特殊标记，后续处理
    else:
        raise ValueError(f"Unsupported fill_type={fill_type!r}")
    return fill_val


def cgpo_v1(images, messages, fill_type='mean'):
    assert len(images) == 1, f"cgpo_v1 currently only supports single image input, but got {len(images)=}"
    img = images[0]
    if isinstance(img, dict):
        img = (img.get('bytes') or img.get('path')) if 'bytes' in img else img['path']
    img = load_image(img)  # PIL.Image.Image

    assert messages[-1]['role'] == 'assistant', f"The last message must be from the assistant."
    response = messages[-1]['content']

    # ---------- 1) 解析response：按id顺序提取三层bbox列表 ----------
    all_bboxes = parse_all_bboxes_from_response(response)

    # ---------- 2) 对每个bbox在原图对应位置做掩模 ----------
    arr = np.array(img)
    H, W = arr.shape[0], arr.shape[1]

    # 计算并填入填充值
    fill_val = get_fill_val(arr, fill_type, all_bboxes)
    if isinstance(fill_val, str) and fill_val in ['local_mean']:
        orig_arr = arr.copy() # 仅在local_mean时使用
    for ent_boxes in all_bboxes:
        for x1, y1, x2, y2 in ent_boxes:
            ax1, ay1, ax2, ay2 = to_abs_bbox(x1, y1, x2, y2, W, H)
            if ax2 <= ax1 or ay2 <= ay1: continue
            if isinstance(fill_val, str) and fill_val in ['local_mean']:
                arr[ay1:ay2, ax1:ax2] = np.round(orig_arr[ay1:ay2, ax1:ax2].mean(axis=(0,1))).astype(arr.dtype) # 确定为Image类型所以arr一定是整数
            else:
                arr[ay1:ay2, ax1:ax2] = fill_val

    img = Image.fromarray(arr)
    return [img]


def cgpo_v2(images, messages, fill_type='mean'):
    assert len(images) == 1, f"cgpo_v2 currently only supports single image input, but got {len(images)=}"
    img = images[0]
    if isinstance(img, dict):
        img = (img.get('bytes') or img.get('path')) if 'bytes' in img else img['path']
    img = load_image(img)  # PIL.Image.Image

    assert messages[-1]['role'] == 'assistant', f"The last message must be from the assistant."
    response = messages[-1]['content']

    # ---------- 1) 解析response：按id顺序提取三层bbox列表 ----------
    all_bboxes = parse_all_bboxes_from_response(response)

    # ---------- 2) 建立bbox层级关系：仅“完全包含”才建立父子（有交集但不包含则忽略） ----------
    arr = np.array(img)
    H, W = arr.shape[0], arr.shape[1]
    orig_arr = arr.copy()  # 混合时必须以“原图”作为基底

    # 扁平化所有bbox为节点（一个bbox一个节点）
    nodes = []  # 每个元素：{ax1,ay1,ax2,ay2,area,parent,children,depth,max_depth,alpha}
    for ent_boxes in all_bboxes:
        for x1, y1, x2, y2 in ent_boxes:
            ax1, ay1, ax2, ay2 = to_abs_bbox(x1, y1, x2, y2, W, H)
            if ax2 <= ax1 or ay2 <= ay1:
                continue
            area = (ax2 - ax1) * (ay2 - ay1)
            nodes.append({
                "ax1": ax1, "ay1": ay1, "ax2": ax2, "ay2": ay2,
                "area": area, "parent": -1, "children": [],
                "depth": -1, "max_depth": -1, "alpha": 1.0,
            })

    def _contains(p, c):
        # 完全包含才算：c完全在p内部/边界内（允许贴边）
        return (p["ax1"] <= c["ax1"] and p["ay1"] <= c["ay1"] and
                p["ax2"] >= c["ax2"] and p["ay2"] >= c["ay2"] and
                p["area"] > c["area"])

    # 为每个节点找“最近父节点”：在所有包含它的候选中选择面积最小的那个
    for i, c in enumerate(nodes):
        best_p, best_area = -1, None
        for j, p in enumerate(nodes):
            if i == j: continue
            if _contains(p, c):
                if best_area is None or p["area"] < best_area:
                    best_area, best_p = p["area"], j
        c["parent"] = best_p
        if best_p != -1:
            nodes[best_p]["children"].append(i)

    # 计算depth（从根到该节点的层级深度）
    def _calc_depth(i):
        if nodes[i]["parent"] == -1:
            return 0
        if nodes[nodes[i]["parent"]]["depth"] == -1:
            nodes[nodes[i]["parent"]]["depth"] = _calc_depth(nodes[i]["parent"])
        return nodes[nodes[i]["parent"]]["depth"] + 1
    
    for i in range(len(nodes)):
        if nodes[i]["depth"] == -1:
            nodes[i]["depth"] = _calc_depth(i)

    # 计算每个节点的max_depth（其子树内最大depth），用于透明度分配
    def _dfs_max(i):
        if nodes[i]["max_depth"] != -1:
            return nodes[i]["max_depth"]
        md = nodes[i]["depth"]
        for ch in nodes[i]["children"]:
            md = max(md, _dfs_max(ch))
        nodes[i]["max_depth"] = md
        return md

    for i in range(len(nodes)):
        if nodes[i]["parent"] == -1:
            _dfs_max(i)

    # 透明度：以该节点子树的最大深度为基准；越外层越“透明”
    for n in nodes:
        n["alpha"] = (n["depth"] + 1) / (n["max_depth"] + 1) # 例如 A(0)->C(1)->F(2): A=1/3, C=2/3, F=1

    # ---------- 3) 掩模：按depth从小到大（先外后内），每层按alpha混合 ----------
    fill_val = get_fill_val(orig_arr, fill_type, all_bboxes)
    use_local = (isinstance(fill_val, str) and fill_val in ['local_mean']) #减少后续字符串比较计算量

    # 先外后内，确保内层（alpha更高）最后覆盖
    nodes.sort(key=lambda d: d["depth"])
    for n in nodes:
        ax1, ay1, ax2, ay2, alpha = n["ax1"], n["ay1"], n["ax2"], n["ay2"], n["alpha"]

        patch_orig = orig_arr[ay1:ay2, ax1:ax2]
        if patch_orig.size == 0:
            continue

        if use_local:
            fv = np.round(patch_orig.mean(axis=(0, 1))).astype(orig_arr.dtype) # 确定为Image类型所以arr一定是整数
        else: # 这里一定不要把fill_val改掉，否则后续所有box会使用第一个box的均值，导致逻辑错误
            fv = fill_val

        # 0为全透明、1为不透明：使用“原图”作为基底进行混合
        if alpha == 1.0:
            arr[ay1:ay2, ax1:ax2] = fv
        else:
            mixed = (1.0 - alpha) * patch_orig.astype(np.float32) + alpha * np.asarray(fv, dtype=np.float32)
            arr[ay1:ay2, ax1:ax2] = np.asarray(np.round(mixed), dtype=orig_arr.dtype)

    img = Image.fromarray(arr)
    return [img]
