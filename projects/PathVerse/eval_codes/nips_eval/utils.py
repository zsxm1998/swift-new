import re
import ast
import math
import json
import xml.etree.ElementTree as ET
from typing import List
from shapely.geometry import Polygon
from shapely.ops import unary_union


def load_json(path):
    with open(path, 'r') as jf:
        data = json.load(jf)
    return data

def dump_json(obj, path, indent=2):
    with open(path, 'w') as jf:
        json.dump(obj, jf, indent=indent, ensure_ascii=False)

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            try:
                data.append(json.loads(line))
            except:
                import sys
                print(f'ERROR: Failed to parse line {i+1} in {file_path}', file=sys.stderr, flush=True)
                raise
    return data

def dump_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')


def think_format_with_suffix(text):
    # 匹配“<think>xxx</think>yyy<answer>zzz</answer>suffix”，其中xxx、zzz不可为空，yyy可为空，suffix可空可有但不能包含标签
    full_pattern = re.compile(
        r'^<think>((?:(?!<think>|</think>|<answer>|</answer>).)+)</think>'   # think 内容：不能有嵌套标签
        r'(.*?)'                                                             # think 和 answer 之间任意内容
        r'<answer>((?:(?!<think>|</think>|<answer>|</answer>).)+)</answer>'  # answer 内容不能包含任何标签符号
        r'(?!.*<think>|.*<answer>|.*</think>|.*</answer>)'                   # 后缀不能再包含标签
        , re.DOTALL
    )

    match = full_pattern.match(text)
    if not match:
        return False

    # 确保 think 和 answer 内容非空
    think_content = match.group(1).strip()
    answer_content = match.group(3).strip()
    if not think_content or not answer_content:
        return False

    # 用于统计是否只出现一次这种结构
    loose_pattern = re.compile(
        r'<think>((?:(?!<think>|</think>|<answer>|</answer>).)+)</think>'
        r'(.*?)'
        r'<answer>((?:(?!<think>|</think>|<answer>|</answer>).)+)</answer>',
        re.DOTALL
    )

    # 确保格式只出现一次
    all_matches = loose_pattern.findall(text)
    if len(all_matches) != 1:
        return False

    return True


def think_format_no_suffix(text):
    # 匹配“<think>xxx</think>yyy<answer>zzz</answer>”，其中xxx、zzz不可为空，yyy可为空
    full_pattern = re.compile(
        r'^<think>((?:(?!<think>|</think>|<answer>|</answer>).)+)</think>'      # think 内容：不能有嵌套标签
        r'(.*?)'                                                                # 任意中间内容
        r'<answer>((?:(?!<think>|</think>|<answer>|</answer>).)+)</answer>$',   # answer 内容：不能有嵌套标签
        re.DOTALL
    )

    match = full_pattern.match(text)
    if not match:
        return False

    # 确保 think 和 answer 内容非空
    think_content = match.group(1).strip()
    answer_content = match.group(3).strip()
    if not think_content or not answer_content:
        return False

    # 用于统计是否只出现一次这种结构
    loose_pattern = re.compile(
        r'<think>((?:(?!<think>|</think>|<answer>|</answer>).)+)</think>'
        r'(.*?)'
        r'<answer>((?:(?!<think>|</think>|<answer>|</answer>).)+)</answer>',
        re.DOTALL
    )

    # 确保格式只出现一次
    all_matches = loose_pattern.findall(text)
    if len(all_matches) != 1:
        return False

    return True


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


def check_other_task_tag_exist(text, stag_set, exclude_tags=[]):
    """测试是否存在其他任务的标签"""
    if isinstance(exclude_tags, str):
        exclude_tags = [exclude_tags]
    for stag in stag_set - set(exclude_tags):
        if stag in text:
            return True
    return False


def has_chinese(text: str) -> bool:
    """判断字符串中是否包含中文字符。"""
    return re.search(r'[\u4e00-\u9fff]', text) is not None


def parse_bbox_string(bbox_str, key_word='box'):
    """
    从形如 <bbox_list><box>160, 312, 254, 414</box>...</bbox_list> 中提取bbox列表
    返回: list of [x1, y1, x2, y2]
    """
    try:
        box_strs = re.findall(rf"<{key_word}>(.*?)</{key_word}>", bbox_str)
        boxes = []
        for box_str in box_strs:
            parts = [int(p.strip()) for p in box_str.strip().split(",")]
            if len(parts) != 4:
                raise ValueError(f"Invalid box: {box_str}")
            if not all(0 <= x <= 1000 for x in parts):
                raise ValueError(f"Coordinates out of range in box: {box_str}")
            boxes.append(parts)
        return boxes
    except Exception as e:
        raise ValueError(f"Failed to parse bbox string: {e}")


def compute_iou(boxA, boxB):
    """计算两个框之间的IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = areaA + areaB - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea


def compute_detection_reward(gt_boxes, pred_boxes):
    """计算基于F1 × avg_IoU 的 reward 值"""
    if not gt_boxes and not pred_boxes:
        return 1.0  # 两者都为空时，reward设为1

    matched_gt = set()
    matched_pred = set()
    match_scores = []

    # 计算所有IoU，并按IoU降序排列
    iou_pairs = []
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou = compute_iou(gt, pred)
            if iou >= 0.5:
                iou_pairs.append((iou, i, j))
    iou_pairs.sort(reverse=True)

    for iou, i, j in iou_pairs:
        if i not in matched_gt and j not in matched_pred:
            matched_gt.add(i)
            matched_pred.add(j)
            match_scores.append(iou)

    TP = len(match_scores)
    FP = len(pred_boxes) - TP
    FN = len(gt_boxes) - TP

    if TP == 0:
        return 0.0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    avg_iou = sum(match_scores) / TP
    reward = f1 * avg_iou
    return reward


def det_no_class_reward(gt_res, content_res):
    """计算不带类别检测任务的reward"""
    if check_negative_exist(gt_res):
        gt_boxes = []
    else:
        gt_boxes = parse_bbox_string(gt_res)
    if check_negative_exist(content_res):
        pred_boxes = []
    else:
        pred_boxes = parse_bbox_string(content_res)
    return compute_detection_reward(gt_boxes, pred_boxes)


def extract_bbox_categories(xml_data):
    """
    提取出每类的 bbox 列表。
    返回字典：{类别: [bbox1, bbox2, ...]}
    """
    # Parse the XML-like structure in the answer field to extract categories and their corresponding bounding boxes
    xml_root = ET.fromstring(xml_data)
    # Extract each category and their bounding boxes
    category_boxes = {}
    for bbox_list in xml_root:
        category = bbox_list.attrib['class']
        boxes = [list(map(int, bbox.text.split(','))) for bbox in bbox_list]
        category_boxes[category] = boxes

    return category_boxes


def det_with_class_reward(gt_res, content_res, alpha=0.8):
    """
    多类别目标检测任务的reward计算，支持混合权重方式。
    
    参数:
        gt_res: 真实的多类别预测结果字符串
        content_res: 模型预测的多类别预测结果字符串
        alpha: float, ∈ [0, 1]，类别均值与数量加权的平衡参数，为1时为类别均值，为0时为数量加权
    
    返回:
        final_reward: float, ∈ [0, 1]
    """
    assert 0.0 <= alpha <= 1.0, "alpha must be between 0 and 1"

    if check_negative_exist(gt_res):
        gt_dict = {}
    else:
        gt_dict = extract_bbox_categories(gt_res)
    if check_negative_exist(content_res):
        pred_dict = {}
    else:
        pred_dict = extract_bbox_categories(content_res)
    
    all_categories = set(gt_dict.keys()) | set(pred_dict.keys())
    if len(all_categories) == 0:
        return 1.0
    reward_per_class = {}
    weight_per_class = {}

    total_gt_boxes = 0

    for category in all_categories:
        gt_boxes = gt_dict.get(category, [])
        pred_boxes = pred_dict.get(category, [])

        reward = compute_detection_reward(gt_boxes, pred_boxes)
        reward_per_class[category] = reward
        weight_per_class[category] = max(len(gt_boxes), 1)  # 至少为1，确保即使gt中无该类，仍有FP惩罚
        total_gt_boxes += len(gt_boxes)

    # 类别平均得分
    mean_reward = sum(reward_per_class.values()) / len(all_categories)

    # 数量加权得分
    total_weight = sum(weight_per_class.values())
    weighted_reward = sum(reward_per_class[c] * weight_per_class[c] for c in all_categories) / total_weight

    # 混合方式
    final_reward = alpha * mean_reward + (1 - alpha) * weighted_reward
    return final_reward


def parse_polygon_string(poly_str):
    """
    解析形如 <contour_list><polygon>[x1, y1], [x2, y2], ...</polygon>...</contour_list>
    的字符串，返回多边形列表
    """
    try:
        # 提取所有 <polygon>...</polygon>
        polygons_raw = re.findall(r"<polygon>(.*?)</polygon>", poly_str)
        polygons = []
        for poly_text in polygons_raw:
            points = ast.literal_eval("[" + poly_text + "]")  # 转为 list of [x, y]
            if not all(isinstance(pt, (list, tuple)) and len(pt) == 2
                       and isinstance(pt[0], int) and isinstance(pt[1], int) for pt in points):
                raise ValueError(f"Invalid polygon points: {poly_text}")
            try:
                polygon = Polygon(points)
                if not polygon.is_valid or polygon.area == 0:
                    continue  # 忽略无效或空区域
            except:
                continue  # 忽略不能构成多边形的点集
            polygons.append(points)

        return polygons

    except Exception as e:
        raise ValueError(f"Failed to parse polygons: {e}")


def seg_reward(gt_res, content_res, beta=0.5):
    """
    输入GT和预测的contour字符串，输出最终reward值（IoU × complexity_penalty）
    """
    if check_negative_exist(gt_res):
        gt_polygons = []
    else:
        gt_polygons = parse_polygon_string(gt_res)
    if check_negative_exist(content_res):
        pred_polygons = []
    else:
        pred_polygons = parse_polygon_string(content_res)
    
    # 构造 shapely MultiPolygon 对象
    gt_shapes = [Polygon(p) for p in gt_polygons]
    pred_shapes = [Polygon(p) for p in pred_polygons]

    if not gt_shapes and not pred_shapes:
        return 1.0  # 都为空，认为是完美匹配
    if not gt_shapes or not pred_shapes:
        return 0.0  # 有一边为空，完全不匹配

    gt_union = unary_union(gt_shapes)
    pred_union = unary_union(pred_shapes)

    intersection = gt_union.intersection(pred_union).area
    union = gt_union.union(pred_union).area
    if union == 0:
        iou = 0.0
    else:
        iou = intersection / union

    # 计算复杂度惩罚
    gt_points = sum(len(p) for p in gt_polygons)
    pred_points = sum(len(p) for p in pred_polygons)
    if pred_points <= gt_points or gt_points == 0:
        penalty = 1.0
    else:
        ratio = pred_points / gt_points
        penalty = math.exp(-beta * (ratio - 1))

    reward = iou * penalty
    return reward
