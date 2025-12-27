import os
import re
import cv2
import json
import numpy as np
from sklearn import metrics
import argparse
import traceback

bbox_questions = [
    "Detect the diagnostic area.",
    "Identify the key area.",
    "Can you detect the diagnostic area in this image?",
    "Please find and mark the key area in this pathology image.",
    "Could you locate and identify the diagnostic areas relevant to the cancer diagnosis?",
    "Analyze the image and detect the key diagnostic areas present.",

    'Does this image have any cancer areas? If so, provide the bounding boxes for each.',
    'Are there cancer regions in this picture? Please give bounding boxes for any cancer areas.',
    'Can you identify cancer in this image? If present, list the bounding boxes of the cancer areas.',
    'Check this image for cancer areas and give me the bounding boxes if there are any.',
    'Is cancer visible in this image? If yes, outline the cancer areas with bounding boxes.',
    'Answer yes or no: Does this pathology image have cancer? If yes, provide bounding boxes for the cancer areas.',
    'Is there cancer in this pathology image? If so, give me the bounding boxes for the cancerous regions.',
    'Can you detect cancer in this pathology image? Yes or no, and if yes, indicate the cancer areas with bounding boxes.',
    'Please confirm whether this pathology image contains cancer. Provide bounding boxes for any cancer areas.',
    'Does this pathology image show any cancer regions? If it does, outline these areas with bounding boxes.',
    'Does this pathology image contain cancer? If so, provide bounding boxes for each area in [x1, y1, x2, y2] format with coordinates normalized between 0 and 1, up to three decimal places.',
    "Is there cancer in this pathology picture? If yes, list the cancer regions' bounding boxes as [x1, y1, x2, y2], with normalized coordinates and three decimal accuracy.",
    'Can you identify cancer areas in this pathology image? Please give their bounding boxes in the format [x1, y1, x2, y2], with normalized 0 to 1 coordinates, precise to three decimals.',
    'Check for cancer in this pathology image and provide the bounding boxes of any found, in the format [x1, y1, x2, y2], with coordinates normalized from 0 to 1 and rounded to three decimal places.',
    'Are there any cancerous regions in this pathology image? If present, outline them using bounding boxes in the format [x1, y1, x2, y2], with normalized coordinates (0 to 1 scale) and three decimal point precision.',
    
    'Does this pathology image contain cancer? If so, provide bounding boxes for each area in [x1, y1, x2, y2] format with coordinates scaled between 0 and 100 as integers.',
    'Is there cancer in this pathology picture? If yes, list the cancer regions\' bounding boxes as [x1, y1, x2, y2], with coordinates scaled between 0 and 100 as integers.',
    'Can you identify cancer areas in this pathology image? Please give their bounding boxes in the format [x1, y1, x2, y2], with coordinates rescaled to 0 to 100 as integers.',
    'Check for cancer in this pathology image and provide the bounding boxes of any found, in the format [x1, y1, x2, y2], with coordinates rescaled from 0 to 100 as integers.',
    'Are there any cancerous regions in this pathology image? If present, outline them using bounding boxes in the format [x1, y1, x2, y2], with coordinates scaled (0 to 100 scale) as integers.',
]
contour_questions = [
    "Segment the diagnostic area.",
    "Can you segment the key area?",
    "Please identify and segment the diagnostic area in this image.",
    "Could you mark and segment the key area in this pathology image?",
    "Can you detect and segment the diagnostic areas relevant to the cancer diagnosis?",
    "Please analyze the image and provide a detailed segmentation of the key diagnostic areas.",

    "Does this image have any cancer areas? If it does, segment out the cancer regions.",
    "Is there cancer in this picture? Please segment the cancerous areas if there are any.",
    "Can you detect any cancer regions in this image? Segment them out if present.",
    "Check this image for cancer. If found, please segment the cancer areas.",
    "Are there any cancerous sections in this image? If yes, extract those areas.",
    "Answer yes or no: Does this pathology image contain cancer areas? If yes, provide their boundaries.",
    "Is there cancer in this pathology image? If so, outline the cancer regions.",
    "Can you confirm if this pathology picture has cancerous areas? If it does, please indicate their edges.",
    "Does this pathology image show any cancer? Yes or no, and if yes, detail the boundaries of the cancer areas.",
    "Are there cancer regions in this pathology image? If present, give the outlines of these areas.",
    "Does this pathology image contain cancer? If yes, provide the boundaries of each cancer area as polygons, with vertex coordinates (x, y) normalized between 0 and 1, accurate to three decimal places.",
    "Is there cancer in this pathology picture? If present, outline each cancer region with a polygon, using normalized coordinates [x, y] up to three decimals.",
    "Can you detect cancer areas in this pathology image? For each, output a boundary polygon with vertices (x, y) normalized to 0-1, with three decimal precision.",
    "Check this pathology image for cancer regions. If found, draw polygons around each area, with vertex coordinates [x, y] normalized and rounded to three decimal places.",
    "Are there any cancerous areas in this pathology image? If so, illustrate their boundaries as polygons, with (x, y) vertices normalized to a range of 0 to 1 and detailed to three decimal points.",

    'Does this pathology image contain cancer? If yes, provide the boundaries of each cancer area as polygons, with vertex coordinates (x, y) rescaled between 0 and 100 as integers.',
    'Is there cancer in this pathology picture? If present, outline each cancer region with a polygon, using rescaled coordinates [x, y] up to 100 as integers.',
    'Can you detect cancer areas in this pathology image? For each, output a boundary polygon with vertices (x, y) scaled to 0-100 as integers.',
    'Check this pathology image for cancer regions. If found, draw polygons around each area, with vertex coordinates [x, y] rescaled and rounded to 100 as integers.',
    'Are there any cancerous areas in this pathology image? If so, illustrate their boundaries as polygons, with (x, y) vertices scaled to a range of 0 to 100 as integers.',
]

# 定义提取block内容的函数
def str2mask(s, n):
    if not s: raise ValueError(f'Wrong s: "{s}""')
    # 创建一个长度为n的全0列表
    result = [0] * n
    # 分割字符串s以获取范围
    ranges = s.split(',')
    for r in ranges:
        if '-' in r:
            # 如果有范围（如'2-4'），将其分割并转换为整数
            start, end = map(int, r.split('-'))
            # 设置相应的索引位置为1
            for i in range(start, end + 1):
                if i < n:  # 确保索引不会超出列表长度
                    result[i] = 1
        else:
            # 如果没有范围（只是单个数字），直接转换并设置为1
            idx = int(r)
            if idx < n:  # 确保索引不会超出列表长度
                result[idx] = 1
    return result

# 定义和测试将模型回答转换为contour/bbox的函数
def remove_outer_tags(input_string):
    start_index = input_string.find('<')
    end_index = input_string.rfind('>')
    if start_index != -1 and end_index != -1:
        return input_string[start_index:end_index + 1]
    else:
        return input_string

def fix_tags(s, keyword='polygon'):
    # 移除连续的闭合标签和连续的开启标签
    s = re.sub(rf'(</{keyword}>)+', f'</{keyword}>', s)
    s = re.sub(rf'(<{keyword}>)+', f'<{keyword}>', s)

    # 分割字符串为标签和文本段
    parts = re.split(rf'(<{keyword}>|</{keyword}>)', s)

    # 初始化修复后的字符串列表
    fixed_parts = []
    open_tag = False

    for part in parts:
        if part == f"<{keyword}>":
            if open_tag:
                # 如果已经有一个打开的 <polygon> 标签，关闭它
                fixed_parts.append(f'</{keyword}>')
            fixed_parts.append(part)
            open_tag = True
        elif part == f"</{keyword}>":
            if open_tag:
                # 只有在标签打开的情况下才添加闭合标签
                fixed_parts.append(part)
                open_tag = False
            else:
                # 忽略无效的闭合标签
                continue
        elif part:
            # 对于文本部分，如果没有打开的 <polygon> 标签，添加一个
            if not open_tag:
                fixed_parts.append(f'<{keyword}>')
                open_tag = True
            fixed_parts.append(part)
            # 如果后面没有立即跟随的闭合标签，添加一个
            if not (parts.index(part) < len(parts) - 1 and parts[parts.index(part) + 1] == f"</{keyword}>"):
                fixed_parts.append(f'</{keyword}>')
                open_tag = False

    # 拼接修复后的字符串
    fixed_string = ''.join(fixed_parts)

    #处理多余的标签
    if fixed_string.endswith(f'<{keyword}>'):
        fixed_string = fixed_string[:-len(f'<{keyword}>')]

    return fixed_string.replace(f'<{keyword}></{keyword}>', '')

def str2list(input_str, outerk='contour_list', innerk='polygon'):
    if input_str.lower().startswith('no') or input_str.lower().startswith('there are no'):
        if f'<{outerk}>' not in input_str and f'<{innerk}>' not in input_str:
            return []
        else:
            raise ValueError(f'Str has No but not empty list: {input_str}')
    corrected_str = input_str.replace('</contourl_list>', '</contour_list>').replace('</contour>', '</polygon>')
    corrected_str = corrected_str.replace('"]', '')
    # Remove irrelevant parts of the string
    corrected_str = remove_outer_tags(corrected_str)

    # Remove <contour_list> and </contour_list>
    if corrected_str.startswith(f'<{outerk}>'):
        corrected_str = corrected_str[len(f'<{outerk}>'):]
    if corrected_str.endswith(f'</{outerk}>'):
        corrected_str = corrected_str[:-len(f'</{outerk}>')]
    # assert f'<{outerk}>' not in corrected_str and f'</{outerk}>' not in corrected_str, f'{input_str}\n\n{corrected_str}'
    if f'</{outerk}>' in corrected_str:
        corrected_str = corrected_str[:corrected_str.index(f'</{outerk}>')]

    # Make sure all <polygon> follow a </polygon>
    corrected_str = fix_tags(corrected_str, keyword=innerk)

    # Add back <contour_list> and </contour_list>
    corrected_str = f'<{outerk}>' + corrected_str + f'</{outerk}>'

    # Format str to list
    corrected_str = corrected_str.replace("'", '').replace('</p>', '').replace('}', '').replace("], 'pink nodule']", '')
    corrected_str = corrected_str.replace('(', '[').replace(')', ']')
    corrected_str = corrected_str.replace('[[', '[').replace(']]', ']')
    corrected_str = corrected_str.replace(f'<{outerk}>', '[').replace(f'</{outerk}>', ']')
    corrected_str = corrected_str.replace(f'</{innerk}><{innerk}>', '], [')
    corrected_str = corrected_str.replace(f'<{innerk}>', '[').replace(f'</{innerk}>', ']')
    corrected_str = corrected_str.replace('] [', '], [')

    # Convert string to list
    try:
        converted_list = eval(corrected_str)
        return converted_list
    except (SyntaxError, TypeError) as e:
        traceback.print_exc()
        print(f"Origin str: {input_str}\n\nConverted str: {corrected_str}\n\nException: {e}")
        return []

# 定义评测和可视化函数
def calculate_iou(box_a, box_b):
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    try:
        iou = intersection_area / float(box_a_area + box_b_area - intersection_area)
    except:
        if box_a[2] == box_a[0] == box_b[2] == box_b[0]:
            return (y_bottom - y_top) / float(max(box_a[3], box_b[3])-min(box_a[1], box_b[1]))
        elif box_a[3] == box_a[1] == box_b[3] == box_b[1]:
            return (x_right - x_left) / float(max(box_a[2], box_b[2])-min(box_a[0], box_b[0]))
        else:
            return 0.0
    return iou

def evaluate_detection(image_info):
    total_true_positives, total_false_positives = 0, 0
    total_false_negatives, total_iou, total_boxes = 0, 0, 0

    for img, info in image_info.items():
        if 'b_true' not in info:
            continue
        try:
            true_boxes = str2list(info['b_true'], outerk='bbox_list', innerk='bbox')
            pred_boxes = str2list(info['b_pred'], outerk='bbox_list', innerk='bbox')
        except Exception as e:
            print(f'{img}:', e)
        matched = set()

        for pred_box in pred_boxes:
            if len(pred_box) != 4:
                continue
            iou_scores = [calculate_iou(pred_box, tb) for tb in true_boxes]
            if iou_scores:
                max_iou = max(iou_scores)
                max_idx = iou_scores.index(max_iou)
                if max_iou > 0.5:
                    total_true_positives += 1
                    total_iou += max_iou
                    matched.add(max_idx)
                else:
                    total_false_positives += 1
            else:
                total_false_positives += 1

        total_false_negatives += len(true_boxes) - len(matched)
        total_boxes += len(true_boxes)

    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    average_iou = total_iou / total_true_positives if total_true_positives > 0 else 0
    accuracy = total_true_positives / total_boxes if total_boxes > 0 else 0 #这里这个accuracy和上面的recall很相近，但不完全相等，因为total_true_positives可能会被重复计数，如果其没有被重复计数，则和recall相等

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "average_iou": average_iou}

def contours_to_mask(contours, shape): #shape=(H, W)
    if isinstance(shape, int):
        shape = (shape, shape)
    mask = np.zeros(shape, dtype=np.uint8)
    for contour in contours:
        # 将归一化坐标转换为实际坐标
        if len(contour) == 0:
            continue
        contour = [c for c in contour if len(c)==2]
        if np.array(contour).dtype == np.float64:
            contour = np.array(contour) * np.array([shape[1], shape[0]]) #这里去掉-1是因为除的时候没-1。np.array([shape[1]-1, shape[0]-1])
        else: # for int results
            contour = np.array(contour) / 100.0 * np.array([shape[1], shape[0]])
        contour = np.round(contour).astype(int)
        cv2.fillPoly(mask, [contour], 1)
    return mask

def evaluate_segmentation(image_info, img_shape):
    metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'Accuracy': []}
    
    for img, info in image_info.items():
        if 'c_true' not in info:
            continue
        c_true = str2list(info['c_true'], outerk='contour_list', innerk='polygon')
        c_pred = str2list(info['c_pred'], outerk='contour_list', innerk='polygon')
        
        mask_true = contours_to_mask(c_true, img_shape)
        mask_pred = contours_to_mask(c_pred, img_shape)
        
        intersection = np.logical_and(mask_true, mask_pred).sum()
        union = np.logical_or(mask_true, mask_pred).sum()
        iou = intersection / union if union else 0
        
        dice = (2. * intersection) / (mask_true.sum() + mask_pred.sum()) if (mask_true.sum() + mask_pred.sum()) else 0
        
        precision = intersection / mask_pred.sum() if mask_pred.sum() else 0
        recall = intersection / mask_true.sum() if mask_true.sum() else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0  #这里的F1就是上面的dice，f1带入P和R后简化就是dice
        
        accuracy = np.sum(mask_true == mask_pred) / mask_true.size if mask_true.size else 0
        
        metrics['IoU'].append(iou)
        metrics['Dice'].append(dice)
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)
        metrics['F1 Score'].append(f1)
        metrics['Accuracy'].append(accuracy)
    
    # 计算平均指标
    for key in metrics.keys():
        metrics[key] = np.mean(metrics[key]) if metrics[key] else 0
    
    return metrics

def visualize_bbox_contour(img_dir, vis_dir, image_info):
    os.makedirs(vis_dir, exist_ok=True)
    for image_file, info in image_info.items():
        if 'b_true' not in info or 'c_true' not in info:
            continue
        img = cv2.imread(os.path.join(img_dir, image_file), cv2.IMREAD_COLOR)

        true_boxes = str2list(info['b_true'], outerk='bbox_list', innerk='bbox')
        pred_boxes = str2list(info['b_pred'], outerk='bbox_list', innerk='bbox')
        true_contours = str2list(info['c_true'], outerk='contour_list', innerk='polygon')
        pred_contours = str2list(info['c_pred'], outerk='contour_list', innerk='polygon')

        # Draw each true bounding box and contour in green
        for box in true_boxes:
            if type(box[0]) == int:
                x1, y1, x2, y2 = [round(coord / 100 * (img.shape[1] if i % 2 == 0 else img.shape[0]))
                                for i, coord in enumerate(box)]
            else:
                x1, y1, x2, y2 = [round(coord * (img.shape[1] if i % 2 == 0 else img.shape[0]))
                                for i, coord in enumerate(box)]
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 192, 0), 10)
        for polygon in true_contours:
            if type(polygon[0][0]) == int:
                poly_points = np.array([[[int(p[0] / 100 * img.shape[1]), int(p[1] / 100 * img.shape[0])]]
                                        for p in polygon], np.int32)
            else:
                poly_points = np.array([[[int(p[0] * img.shape[1]), int(p[1] * img.shape[0])]]
                                        for p in polygon], np.int32)
            cv2.polylines(img, [poly_points], True, (0, 255, 0), 10)
        # Draw each pred bounding box and contour in red
        for box in pred_boxes:
            if len(box) != 4:
                continue
            if type(box[0]) == int:
                x1, y1, x2, y2 = [round(coord / 100 * (img.shape[1] if i % 2 == 0 else img.shape[0]))
                                for i, coord in enumerate(box)]
            else:
                x1, y1, x2, y2 = [round(coord * (img.shape[1] if i % 2 == 0 else img.shape[0]))
                                for i, coord in enumerate(box)]
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 0, 192), 10)
        for polygon in pred_contours:
            polygon = [c for c in polygon if len(c)==2]
            if type(polygon[0][0]) == int:
                poly_points = np.array([[[int(p[0] / 100 * img.shape[1]), int(p[1] / 100 * img.shape[0])]]
                                        for p in polygon], np.int32)
            else:
                poly_points = np.array([[[int(p[0] * img.shape[1]), int(p[1] * img.shape[0])]]
                                        for p in polygon], np.int32)
            cv2.polylines(img, [poly_points], True, (0, 0, 255), 10)

        # Save image
        cv2.imwrite(os.path.join(vis_dir, image_file.replace('/', '_')), img)

def evaluate_subtype(image_info):
    true_list, pred_list = [], []
    for img, info in image_info.items():
        try:
            true_list.append(info['g_true'].replace('.','').lower())
            pred_list.append(info['g_pred'].replace('.','').lower())
        except:
            print(f'Exception at {img}:')
            raise
    assert len(true_list) == len(pred_list)
    all_labels = sorted(set(true_list))
    print('Liver cancer thumbnails classification results:')
    print(f'True category number: {len(set(true_list))}, Pred category number: {len(set(pred_list))}')
    print(metrics.classification_report(true_list, pred_list, digits=4, zero_division=0, labels=all_labels, target_names=all_labels))
    print('Accuracy:', round(metrics.accuracy_score(true_list, pred_list), 4))
    print('Balanced ACC:', round(metrics.balanced_accuracy_score(true_list, pred_list), 4))


def main(img_dir, q_file, a_file, vis_dir):
    qjsons, ajsons = [], []
    with open(q_file, 'r') as file:
        for line in file: qjsons.append(json.loads(line))
    with open(a_file, 'r') as file:
        for line in file: ajsons.append(json.loads(line))

    image_info = {}
    for tr, pr in zip(qjsons, ajsons):
        assert tr['question_id'] == pr['question_id'], f"{tr['question_id']} vs {pr['question_id']}"
        if tr['text'] in bbox_questions:
            image_info.setdefault(tr['image'], {})['b_true'] = tr['answer']
            image_info.setdefault(tr['image'], {})['b_pred'] = pr['text']
        elif tr['text'] in contour_questions:
            image_info.setdefault(tr['image'], {})['c_true'] = tr['answer']
            image_info.setdefault(tr['image'], {})['c_pred'] = pr['text']

    # 评测bbox的性能
    dec_res = evaluate_detection(image_info)
    print(f"detection accuracy: {dec_res['accuracy']*100:.2f}, precision: {dec_res['precision']:.4f}", end=', ')
    print(f"recall: {dec_res['recall']:.4f}, f1-score: {dec_res['f1']:.4f}, average_iou: {dec_res['average_iou']:.4f}")

    # 评测contour分割的性能
    seg_res = evaluate_segmentation(image_info, img_shape=336)
    print(f"\nsegmentation IoU: {seg_res['IoU']:.4f}, Dice: {seg_res['Dice']:.4f}, Pixel Accuracy:{seg_res['Accuracy']:.4f}")
    print(f"\tPrecision: {seg_res['Precision']:.4f}, Recall: {seg_res['Recall']:.4f}, F1 Score: {seg_res['F1 Score']:.4f}")

    # 可视化bbox和contour
    if vis_dir:
        visualize_bbox_contour(img_dir, vis_dir, image_info)

def get_args():
    parser = argparse.ArgumentParser(description='evaluate liverWSI',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_dir', type=str, default="", help='path to image folders')
    parser.add_argument('--q_file', type=str, default="", help='path to groundtruth file')
    parser.add_argument('--a_file', type=str, default="", help='path to prediction file')
    parser.add_argument('--vis_dir', type=str, default=None, help='path to visualize folders')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args.img_dir, args.q_file, args.a_file, args.vis_dir)