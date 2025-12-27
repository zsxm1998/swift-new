import os
import re
import cv2
import json
import argparse
import traceback

bbox_questions = [
    # detect nucleus
    'Please identify all nuclei in this image.',
    'Detect every cell nucleus present in the picture.',
    'Identify all the nuclei within this image.',
    'Find and mark all nuclei in the image.',
    'Locate every nucleus in this picture.',
    'Detect all cell nuclei in the image using bounding boxes.',
    'Identify every nucleus in the picture and mark them with bbox.',
    'Please use bbox to outline all nuclei present in this image.',
    'Find all the cell nuclei in the image and use bounding boxes for each.',
    'Locate and mark every nucleus in the picture with a bbox.',
    'Detect all nuclei in this pathology image and output with bounding boxes in [x1, y1, x2, y2] format, normalized coordinates to 0-1, accurate to three decimals.',
    'Identify every cell nucleus in the picture, marking them with bbox in [x1, y1, x2, y2], normalize coordinates between 0 and 1, with three decimal precision.',
    'Please use bbox to indicate all nuclei in this image, with coordinates in [x1, y1, x2, y2] format, normalized to 0-1 and rounded to three decimal places.',
    'Find all nuclei in the pathology image and represent each with a bounding box, using [x1, y1, x2, y2] for normalized coordinates to a scale of 0 to 1, with three digits after the decimal.',
    'Locate every nucleus in this image, using bbox for output in [x1, y1, x2, y2] format, with coordinates normalized to 0-1, and precision up to three decimals.',
    
    'Detect all nuclei in this pathology image and output with bounding boxes in [x1, y1, x2, y2] format, with coordinates scaled to 0-100 as integers.',
    'Identify every cell nucleus in the picture, marking them with bbox in [x1, y1, x2, y2], with coordinates rescaled between 0 and 100 as integers.',
    'Please use bbox to indicate all nuclei in this image, with coordinates in [x1, y1, x2, y2] format, scaled to 0-100 as integers.',
    'Find all nuclei in the pathology image and represent each with a bounding box, using [x1, y1, x2, y2] for coordinates rescaled to a scale of 0 to 100 as integers.',
    'Locate every nucleus in this image, using bbox for output in [x1, y1, x2, y2] format, with coordinates scaled to 0-100 as integers.',
    
    # detect vessel
    "Detect all vessels",
    "Find every blood vessel.",
    "Identify all vessels in image",
    "Locate all blood vessels.",
    "Can you detect all blood vessels in this image?",
    "Could you show all the vessels in the image?",
    "Locate and mark every blood vessel in this picture?",
    "Please identify and create bounding boxes around every blood vessel visible in this image, including both large and small vessels.",
    
    # MVI cancerous nuclei detection
    'Please identify all cancerous nuclei in this vessel.',
    'Detect every cancerous cell nucleus present in the vessel.',
    'Identify all the cancerous nuclei within this blood vessel.',
    'Find and mark all cancerous cell nuclei in the vessel.',
    'Locate every cancerous nucleus in this blood vessel.',
    'Detect all cancerous nuclei in the vessel using bounding boxes.',
    'Identify every cancerous nucleus in the vessel and mark them with bbox.',
    'Please use bbox to outline all cancerous cell nuclei present in this vessel.',
    'Find all the cancerous nuclei in the vessel and use bounding boxes for each.',
    'Locate and mark every cancerous cell nucleus in the blood vessel with a bbox.',

    "Count and locate cancerous nuclei in the vessel",
    "Cancerous nucleus count and locations in the blood vessel?",
    "Count cancerous nuclei and mark locations in the vessel.",
    "How many cancerous nuclei are there and where are they in the blood vessel?",
    "Can you count the cancerous cell nuclei in the vessel and provide their locations?",
    "Can you identify and count all cancerous nuclei in the blood vessel, and indicate their locations?",
    "Please provide a detailed count of all the cancerous cell nuclei present in this vessel, along with the exact locations of each nucleus.",

    "Detect and count cancerous nuclei in the vessel",
    "Detect cancerous cell nuclei in the blood vessel and count.",
    "Find cancerous nuclei in the vessel and give number?",
    "Detect all cancerous nuclei in the blood vessel and count them.",
    "Can you identify and count all the cancerous nuclei in this vessel?",
    "Please detect every cancerous cell nucleus in the vessel and provide a total count.",
    "Could you perform a comprehensive detection of cancerous cell nuclei within this blood vessel and accurately report their total number?",

    # LN det
    "Detect all lymph nodes.",
    # nerve det
    "Detect all nerves.",

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

def str2list(input_str, outerk='bbox_list', innerk='box'):
    if input_str.lower().startswith('no'):
        if f'<{outerk}>' not in input_str and f'<{innerk}>' not in input_str:
            return []
        else:
            raise ValueError(f'Str has No but not empty list: {input_str}')
    corrected_str = input_str.replace('</contourl_list>', '</contour_list>').replace('</contour>', '</polygon>')
    corrected_str = corrected_str.replace('<detection_result><bbox_list class="inflammatory cell">', f'<{outerk}>')
    corrected_str = re.sub(r'<detection_result><bbox_list class="[^"]*">', f'<{outerk}>', corrected_str)
    corrected_str = corrected_str.replace('</detection_result>', f'</{outerk}>')
    
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
        try:
            true_boxes = str2list(info['b_true'], outerk='bbox_list', innerk='bbox')
            pred_boxes = str2list(info['b_pred'], outerk='bbox_list', innerk='box')
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


def visualize_bbox_contour(img_dir, vis_dir, image_info):
    os.makedirs(vis_dir, exist_ok=True)
    for image_file, info in image_info.items():
        img = cv2.imread(os.path.join(img_dir, image_file), cv2.IMREAD_COLOR)

        true_boxes = str2list(info['b_true'], outerk='bbox_list', innerk='bbox')
        pred_boxes = str2list(info['b_pred'], outerk='bbox_list', innerk='box')

        # Draw each true bounding box in green
        for box in true_boxes:
            if type(box[0]) == int:
                x1, y1, x2, y2 = [round(coord / 100 * (img.shape[1] if i % 2 == 0 else img.shape[0]))
                                for i, coord in enumerate(box)]
            else:
                x1, y1, x2, y2 = [round(coord * (img.shape[1] if i % 2 == 0 else img.shape[0]))
                                for i, coord in enumerate(box)]
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 192, 0), 2)
        # Draw each pred bounding box in red
        for box in pred_boxes:
            if len(box) != 4:
                continue
            if type(box[0]) == int:
                x1, y1, x2, y2 = [round(coord / 100 * (img.shape[1] if i % 2 == 0 else img.shape[0]))
                                for i, coord in enumerate(box)]
            else:
                x1, y1, x2, y2 = [round(coord * (img.shape[1] if i % 2 == 0 else img.shape[0]))
                                for i, coord in enumerate(box)]
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 0, 192), 2)

        # Save image
        cv2.imwrite(os.path.join(vis_dir, image_file.replace('.tif', '.png').replace('/', '-')), img)

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

    # 评测bbox的性能
    dec_res = evaluate_detection(image_info)
    print(f"detection precision: {dec_res['precision']:.4f}", end=', ')
    print(f"recall: {dec_res['recall']:.4f}, f1-score: {dec_res['f1']:.4f}, average_iou: {dec_res['average_iou']:.4f}")

    # 可视化bbox和contour
    if vis_dir:
        visualize_bbox_contour(img_dir, vis_dir, image_info)

def get_args():
    parser = argparse.ArgumentParser(description='evaluate no class detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_dir', type=str, default="", help='path to image folders')
    parser.add_argument('--q_file', type=str, default="", help='path to groundtruth file')
    parser.add_argument('--a_file', type=str, default="", help='path to prediction file')
    parser.add_argument('--vis_dir', type=str, default=None, help='path to visualize folders')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args.img_dir, args.q_file, args.a_file, args.vis_dir)