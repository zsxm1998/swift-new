import os
import re
import cv2
import json
import numpy as np
from sklearn import metrics
import argparse
import traceback

contour_questions = [
    # datasets/ZheYi0607/liver_cancer/thumbnails
    "Segment the diagnostic area.",
    "Can you segment the key area?",
    "Please identify and segment the diagnostic area in this image.",
    "Could you mark and segment the key area in this pathology image?",
    "Can you detect and segment the diagnostic areas relevant to the cancer diagnosis?",
    "Please analyze the image and provide a detailed segmentation of the key diagnostic areas.",
    # datasets/ZheYi0607/LNM
    "Segment the cancerous area in the lymph node.",
    "Can you segment the cancerous region in this lymph node?",
    "Please identify and segment the cancerous areas within this lymph node.",
    "Could you analyze and segment all the cancerous regions in the lymph node shown in this image?",
    # datasets/ZheYi0607/NI_cls
    "Segment the cancerous area in the nerve.",
    "Can you segment the cancerous region in this nerve?",
    "Please identify and segment the cancerous areas within this nerve.",
    "Could you analyze and segment all the cancerous regions in the nerve shown in this image?",
    "Can you detect and segment the specific areas of cancer within the nerve in this pathology image?",
    # datasets/ZheYi0730/NI_det and datasets/ZheYi0730/NI_cls
    'Segment all nerves.',
]

seg_healthy_nerve = [
    "Segment all nerves.",
    "Can you segment the nerves in this image?",
    "Identify and segment all nerves present in the pathology image.",
    "Please detect and segment all nerves in this pathology slide.",
    "Could you locate, identify, and segment every nerve visible in this pathology image?",
]
seg_cancer_in_nerve = [
    "Segment the cancerous area in the nerve.",
    "Can you segment the cancerous region in this nerve?",
    "Please identify and segment the cancerous areas within this nerve.",
    "Could you analyze and segment all the cancerous regions in the nerve shown in this image?",
    "Can you detect and segment the specific areas of cancer within the nerve in this pathology image?",
]

q_list_dict = {
    'HN_seg': seg_healthy_nerve,
    'NC_seg': seg_cancer_in_nerve, 
}

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

def visualize_contour(img_dir, vis_dir, image_info):
    os.makedirs(vis_dir, exist_ok=True)
    for image_file, info in image_info.items():
        img = cv2.imread(os.path.join(img_dir, image_file), cv2.IMREAD_COLOR)

        true_contours = str2list(info['c_true'], outerk='contour_list', innerk='polygon')
        pred_contours = str2list(info['c_pred'], outerk='contour_list', innerk='polygon')

        # Draw each true contour in green
        for polygon in true_contours:
            if type(polygon[0][0]) == int:
                poly_points = np.array([[[int(p[0] / 100 * img.shape[1]), int(p[1] / 100 * img.shape[0])]]
                                        for p in polygon], np.int32)
            else:
                poly_points = np.array([[[int(p[0] * img.shape[1]), int(p[1] * img.shape[0])]]
                                        for p in polygon], np.int32)
            cv2.polylines(img, [poly_points], True, (0, 255, 0), 10)
        # Draw each pred contour in red
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


def main(img_dir, q_file, a_file, vis_dir, question_list):
    qjsons, ajsons = [], []
    with open(q_file, 'r') as file:
        for line in file: qjsons.append(json.loads(line))
    with open(a_file, 'r') as file:
        for line in file: ajsons.append(json.loads(line))

    image_info = {}
    for tr, pr in zip(qjsons, ajsons):
        assert tr['question_id'] == pr['question_id'], f"{tr['question_id']} vs {pr['question_id']}"
        if tr['text'] in question_list:
            image_info.setdefault(tr['image'], {})['c_true'] = tr['answer']
            image_info.setdefault(tr['image'], {})['c_pred'] = pr['text']

    # 评测contour分割的性能
    seg_res = evaluate_segmentation(image_info, img_shape=336)
    print(f"segmentation IoU: {seg_res['IoU']:.4f}, Dice: {seg_res['Dice']:.4f}, Pixel Accuracy:{seg_res['Accuracy']:.4f}")
    print(f"\tPrecision: {seg_res['Precision']:.4f}, Recall: {seg_res['Recall']:.4f}, F1 Score: {seg_res['F1 Score']:.4f}")

    # 可视化contour
    if vis_dir:
        visualize_contour(img_dir, vis_dir, image_info)

def get_args():
    parser = argparse.ArgumentParser(description='evaluate no class segmentation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_dir', type=str, default="", help='path to image folders')
    parser.add_argument('--q_file', type=str, default="", help='path to groundtruth file')
    parser.add_argument('--a_file', type=str, default="", help='path to prediction file')
    parser.add_argument('--vis_dir', type=str, default=None, help='path to visualize folders')
    parser.add_argument('--dataset', type=str, default=None, help='key to find question list')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    q_list = contour_questions if args.dataset is None else q_list_dict[args.dataset]
    main(args.img_dir, args.q_file, args.a_file, args.vis_dir, q_list)