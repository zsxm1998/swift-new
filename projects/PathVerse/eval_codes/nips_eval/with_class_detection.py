import os
import os.path as osp
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import colorsys

from no_class_detection import get_dataset
from utils import compute_iou, extract_bbox_categories


question_lists = {
    'nucleus': [
        'Please detect and classify all nuclei in this image.',
        'Detect and classify every cell nucleus present in the picture.',
        'Detect and classify all the nuclei within this image.',
        'Detect and classify all nuclei in the image.',
        'Locate every nucleus and give its category in this picture.',
        'Detect all cell nuclei in the image using bounding boxes with labels.',
        'Detect and classify every nucleus in the picture and mark them with bbox.',
        'Please use bbox to outline all nuclei and indicate every label present in this image.',
        'Distinguish all the cell nuclei in the image and use bounding boxes for each.',
        'Detect and classify every nucleus in the picture with a bbox.',
        'Detect and classify all nuclei in this pathology image and output with bounding boxes in [x1, y1, x2, y2] format, with coordinates scaled to 0-1000 as integers.',
        'Identify every cell nucleus with label in the picture, marking them with bbox in [x1, y1, x2, y2], with coordinates scaled between 0 and 1000 as integers.',
        'Please use bbox to detect and classify all nuclei in this image, with coordinates in [x1, y1, x2, y2] format, scaled to 0-1000 as integers.',
        'Find all nuclei in the pathology image and represent each with a bounding box and a category, using [x1, y1, x2, y2] for coordinates scaled to a scale of 0 to 1000 as integers.',
        'Locate and classify every nucleus in this image, using bbox for output in [x1, y1, x2, y2] format, with coordinates scaled to 0-1000 as integers.',
    ],
    'neural_invasion': [
        'Detect all nerves and classify for invasion.', # NI_det 0730
    ],
}


def set_chinese_font_global():
    """设置全局中文字体，避免每次手动指定 fontproperties。"""
    preferred_fonts = [
        "WenQuanYi Zen Hei",
        "Noto Sans CJK JP",
        "SimHei",
    ]
    for font_name in preferred_fonts:
        try:
            fm.findfont(font_name, fallback_to_default=False)  # 检查字体是否存在
            plt.rcParams['font.sans-serif'] = [font_name]       # 成功就设置
            plt.rcParams['axes.unicode_minus'] = False
            print(f"[Info] 中文字体已设置为: {font_name}", file=sys.stderr)
            return
        except:
            continue
    print("[Warning] 未找到合适的中文字体，中文可能无法正常显示。", file=sys.stderr)


def evaluate_multi_class_detection(data_list, iou_threshold=0.5):
    # 初始化统计容器
    class_stats = {}         # 按类别统计
    novel_FP = 0             # 新类别（GT中不存在的类别）的FP总数
    num_negative_images = 0  # 全类别负样本图片数（图片上所有类别均无GT，即空图）
    negative_TN = 0          # 全类别正确拒绝数（图片上所有类别均无预测）
    negative_FP = 0          # 全类别误检数（图片上至少一个类别有预测）

    # ================================ 先统计所有合法类别 ================================
    for _, _, gt_dict, _ in data_list:
        for cls in gt_dict.keys():
            if cls not in class_stats:
                class_stats[cls] = {'TP': 0, 'FP': 0, 'FN': 0, 'IoU': 0}

    # ================================ 遍历数据统计各类指标 ================================
    for _, _, gt_dict, pred_dict in data_list:
        # -------------------------------- 全类别负样本（空图）统计 --------------------------------
        all_gt_empty = all(len(v) == 0 for v in gt_dict.values()) # 此处gt_dict为空也为True
        if all_gt_empty:
            num_negative_images += 1
            all_pred_empty = all(len(v) == 0 for v in pred_dict.values())
            if all_pred_empty:
                negative_TN += 1
            else:
                negative_FP += sum(len(preds) for preds in pred_dict.values())
        
        # -------------------------------- 逐类别处理 --------------------------------
        all_classes = set(gt_dict.keys()).union(pred_dict.keys())
        
        for cls in all_classes:
            gt_boxes = gt_dict.get(cls, [])
            pred_boxes = pred_dict.get(cls, [])
            
            # -------------------------------- 未见类别处理 --------------------------------
            if cls not in class_stats:
                novel_FP += len(pred_boxes)
                continue
                
            # -------------------------------- 常规类别匹配 --------------------------------
            matched_gt, matched_pred, match_scores = set(), set(), []
            
            # 生成IoU矩阵并匹配
            iou_pairs = []
            for i, gt in enumerate(gt_boxes):
                for j, pred in enumerate(pred_boxes):
                    try:
                        iou = compute_iou(gt, pred)
                    except:
                        iou = 0.0
                    if iou >= iou_threshold:
                        iou_pairs.append( (iou, i, j) )
            
            # 按IoU降序匹配（优先匹配高IoU对）
            iou_pairs.sort(reverse=True)
            for iou, i, j in iou_pairs:
                if i not in matched_gt and j not in matched_pred:
                    matched_gt.add(i)
                    matched_pred.add(j)
                    match_scores.append(iou)
            
            # 更新类别统计量
            TP = len(match_scores)
            class_stats[cls]['TP'] += TP
            class_stats[cls]['FP'] += len(pred_boxes) - TP
            class_stats[cls]['FN'] += len(gt_boxes) - TP
            class_stats[cls]['IoU'] += sum(match_scores)

    # ================================ 指标计算 ================================
    
    # -------------------------------- 按类别统计 --------------------------------
    per_class_metrics = {}
    for cls in class_stats:
        TP, FP, FN = class_stats[cls]['TP'], class_stats[cls]['FP'], class_stats[cls]['FN']
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        average_iou = class_stats[cls]['IoU'] / TP if TP > 0 else 0.0
        
        per_class_metrics[cls] = {'P': precision, 'R': recall, 'F1': f1, 'avg_IoU': average_iou}
    
    # -------------------------------- Macro平均 --------------------------------
    macro_precision = np.mean([v['P'] for v in per_class_metrics.values()])
    macro_recall = np.mean([v['R'] for v in per_class_metrics.values()])
    macro_f1 = np.mean([v['F1'] for v in per_class_metrics.values()])
    macro_avg_iou = np.mean([v['avg_IoU'] for v in per_class_metrics.values()])
    
    # -------------------------------- Micro平均 --------------------------------
    total_TP = sum(v['TP'] for v in class_stats.values())
    total_FP = sum(v['FP'] for v in class_stats.values()) + novel_FP
    total_FN = sum(v['FN'] for v in class_stats.values())
    total_iou = sum(v['IoU'] for v in class_stats.values())
    
    micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    micro_avg_iou = total_iou / total_TP if total_TP > 0 else 0.0
    
    # -------------------------------- 未见类别指标 --------------------------------
    novel_FP_rate = novel_FP / total_FP if total_FP > 0 else 0.0
    
    # -------------------------------- 空图指标 --------------------------------
    negative_FPR = negative_FP / num_negative_images if num_negative_images > 0 else 0.0
    negative_TNR = negative_TN / num_negative_images if num_negative_images > 0 else 0.0
    
    return {
        'Per-Class': per_class_metrics,
        'Macro': {'P': macro_precision, 'R': macro_recall, 'F1': macro_f1, 'avg_IoU': macro_avg_iou},
        'Micro': {'P': micro_precision, 'R': micro_recall, 'F1': micro_f1, 'avg_IoU': micro_avg_iou},
        'Novel_FP': novel_FP,
        'Novel_FP_Rate': novel_FP_rate,
        'Negative_FPR': negative_FPR,
        'Negative_TNR': negative_TNR
    }


def generate_distinct_colors(n):
    hsv_colors = [(i / n, 0.6, 0.9) for i in range(n)]  # hue 均匀，固定中高饱和度+亮度
    rgb_colors = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors]
    return rgb_colors


def visualize_bbox(vis_dir, data_list):
    os.makedirs(vis_dir, exist_ok=True)

    # ========== Step 1: 收集所有类别 ==========
    all_classes = set()
    for _, _, gt_dict, pred_dict in data_list:
        all_classes.update(gt_dict.keys())
        all_classes.update(pred_dict.keys())
    all_classes = sorted(list(all_classes))

    # ========== Step 2: 为每个类别生成颜色 ==========
    distinct_colors = generate_distinct_colors(len(all_classes))
    color_map = {cls: rgb for cls, rgb in zip(all_classes, distinct_colors)}

    for image_path, _, gt_dict, pred_dict in data_list:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f'Warning: failed to load {image_path}')
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        # ========== Step 3: 创建子图，左GT，右Pred ==========
        fig, axs = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1, 0.2]})
        ax_gt, ax_pred, ax_legend = axs
        ax_gt.set_title("GT")
        ax_pred.set_title("Prediction")

        for ax in [ax_gt, ax_pred]:
            ax.imshow(img)
            ax.axis('off')

        # ========== Step 4: 可视化 bbox ==========
        visible_classes = set()

        def draw_boxes(ax, box_dict):
            for cls, boxes in box_dict.items():
                color = color_map[cls]
                for box in boxes:
                    x1, y1, x2, y2 = [int(coord / 1000 * W) if i % 2 == 0 else int(coord / 1000 * H) for i, coord in enumerate(box)]
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, cls, fontsize=8, color=color, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
                if boxes:
                    visible_classes.add(cls)

        draw_boxes(ax_gt, gt_dict)
        draw_boxes(ax_pred, pred_dict)

        # ========== Step 5: 绘制图例 ==========
        ax_legend.axis('off')
        y_offset = 1.0
        for cls in sorted(visible_classes):
            color = color_map[cls]
            ax_legend.add_patch(patches.Rectangle((0, y_offset - 0.05), 0.2, 0.04, color=color))
            ax_legend.text(0.25, y_offset - 0.03, cls, transform=ax_legend.transAxes, fontsize=10)
            y_offset -= 0.07

        # ========== Step 6: 保存图像 ==========
        save_path = osp.join(vis_dir, osp.splitext(osp.basename(image_path))[0] + '.jpg')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()


def main(args):
    all_questions = {question for sublist in question_lists.values() for question in sublist}
    dataset = get_dataset(args, parse_func=extract_bbox_categories, stag='<detection_result>',
                          etag='</detection_result>', null_func=dict, valid_questions=all_questions)

    results_dict = {key: [] for key in question_lists}
    unknown_questions = set()
    for data in dataset:
        question = data[1]
        for data_type, prompt_list in question_lists.items():
            if question in prompt_list:
                results_dict[data_type].append(data)
                break
        else:
            if args.report_unknown_question and question not in unknown_questions:
                print(f'Unknown question: {question}')
                unknown_questions.add(question)

    only_one_data_type = len([x for x in results_dict.values() if x]) == 1
    for data_type, data_list in results_dict.items():
        if not data_list:
            continue
        print(f'---------------------[{data_type}]---------------------')
        res = evaluate_multi_class_detection(data_list)

        print('Per-Class:')
        for cls, m in res['Per-Class'].items():
            print(f'  {cls}: Precision: {m["P"]*100:.2f}, Recall: {m["R"]*100:.2f}, F1-score: {m["F1"]*100:.2f}, Average IoU: {m["avg_IoU"]*100:.2f}')
        
        print('Macro average:')
        m = res['Macro']
        print(f'  Precision: {m["P"]*100:.2f}, Recall: {m["R"]*100:.2f}, F1-score: {m["F1"]*100:.2f}, Average IoU: {m["avg_IoU"]*100:.2f}')

        print('Micro average:')
        m = res['Micro']
        print(f'  Precision: {m["P"]*100:.2f}, Recall: {m["R"]*100:.2f}, F1-score: {m["F1"]*100:.2f}, Average IoU: {m["avg_IoU"]*100:.2f}')

        print(f'Novel FP: {res["Novel_FP"]}, Novel FP Rate: {res["Novel_FP_Rate"]*100:.2f}', end='    ')
        print(f'Negative FPR: {res["Negative_FPR"]*100:.2f}, Negative TNR: {res["Negative_TNR"]*100:.2f}', end='\n\n')

        if args.vis_dir:
            vis_dir = args.vis_dir if only_one_data_type else osp.join(args.vis_dir, data_type)
            visualize_bbox(vis_dir, data_list)


if __name__ == '__main__':
    set_chinese_font_global()
    parser = argparse.ArgumentParser(description='evaluate no class detection', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_file', type=str, required=True, help='path to the generation result jsonl file')
    parser.add_argument('--gt_file', type=str, default=None, help='path to groundtruth file')
    parser.add_argument('--dataset', type=str, default=None, help='only evaluate images of which path contains the dataset name')
    parser.add_argument('--img_dir', type=str, default=None, help='path to root image folder')
    parser.add_argument('--vis_dir', type=str, default=None, help='path to visualize folder')
    parser.add_argument('--report_unknown_question', action='store_true', help='report unknown question')
    args = parser.parse_args()
    if args.dataset is not None: # for fnmatch
        args.dataset = '*' + args.dataset + '*'
    main(args)