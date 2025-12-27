import os
import sys
import json
import xml.etree.ElementTree as ET
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import traceback


def extract_bbox_categories(xml_data):
    # Parse the XML-like structure in the answer field to extract categories and their corresponding bounding boxes
    xml_root = ET.fromstring(xml_data)
    # Extract each category and their bounding boxes
    category_boxes = {}
    for bbox_list in xml_root:
        category = bbox_list.attrib['class']
        boxes = [list(map(float, bbox.text.split(','))) for bbox in bbox_list]
        category_boxes[category] = boxes

    return category_boxes


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
        try:
            if box_a[2] == box_a[0] == box_b[2] == box_b[0]:
                return (y_bottom - y_top) / float(max(box_a[3], box_b[3])-min(box_a[1], box_b[1]))
            elif box_a[3] == box_a[1] == box_b[3] == box_b[1]:
                return (x_right - x_left) / float(max(box_a[2], box_b[2])-min(box_a[0], box_b[0]))
            else:
                return 0.0
        except:
            return 0.0
    return iou


def metrics_per_class(true_boxes, pred_boxes, iou_threshold=0.5):
    """Calculate precision, recall, F1-score, and average IoU for a class."""
    tp = 0
    fp = 0
    fn = 0
    total_iou = 0
    matches = []
    # Check each prediction for a match with the true boxes
    for pred in pred_boxes:
        if len(pred) != 4:
            continue
        matched = False
        for true in true_boxes:
            iou = calculate_iou(pred, true)
            if iou >= iou_threshold:
                if true not in matches:
                    matches.append(true)
                    tp += 1
                    total_iou += iou
                    matched = True
                    break
        if not matched:
            fp += 1

    # All unmatched true boxes are false negatives
    fn = len(true_boxes) - len(matches)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = total_iou / tp if tp > 0 else 0

    return precision, recall, f1_score, avg_iou


def calculate_metrics(b_trues, b_preds):
    """Calculate and average the metrics over all classes."""
    class_metrics = {}
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_ious = []

    # Process each class and each image
    for true, pred in zip(b_trues, b_preds):
        for class_name in true:
            true_boxes = true[class_name]
            pred_boxes = pred.get(class_name, [])
            precision, recall, f1, iou = metrics_per_class(true_boxes, pred_boxes)

            # Store individual class metrics
            if class_name in class_metrics:
                class_metrics[class_name]['precision'].append(precision)
                class_metrics[class_name]['recall'].append(recall)
                class_metrics[class_name]['f1_score'].append(f1)
                class_metrics[class_name]['iou'].append(iou)
            else:
                class_metrics[class_name] = {
                    'precision': [precision],
                    'recall': [recall],
                    'f1_score': [f1],
                    'iou': [iou]
                }

            # Store overall metrics
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)
            all_ious.append(iou)

    # Average the metrics per class
    for class_name, metrics in class_metrics.items():
        class_metrics[class_name] = {
            'precision': sum(metrics['precision']) / len(metrics['precision']),
            'recall': sum(metrics['recall']) / len(metrics['recall']),
            'f1_score': sum(metrics['f1_score']) / len(metrics['f1_score']),
            'iou': sum(metrics['iou']) / len(metrics['iou'])
        }

    # Calculate the average metrics across all classes
    average_metrics = {
        'precision': sum(all_precisions) / len(all_precisions),
        'recall': sum(all_recalls) / len(all_recalls),
        'f1_score': sum(all_f1_scores) / len(all_f1_scores),
        'iou': sum(all_ious) / len(all_ious)
    }

    return class_metrics, average_metrics


def convert_to_absolute(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    # Check if the coordinates are in 0-100 range
    if max(x1, x2, y1, y2) > 1:
        x1 = x1 / 100 * img_width
        y1 = y1 / 100 * img_height
        x2 = x2 / 100 * img_width
        y2 = y2 / 100 * img_height
    else:  # Coordinates are in 0-1 range
        x1 = x1 * img_width
        y1 = y1 * img_height
        x2 = x2 * img_width
        y2 = y2 * img_height
    return [x1, y1, x2, y2]


def draw_boxes(ax, data, img_width, img_height, edge_color, label, place_label_bottom=False):
    for bbox in data:
        if len(bbox) != 4:
            continue
        abs_bbox = convert_to_absolute(bbox, img_width, img_height)
        x1, y1, x2, y2 = abs_bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=edge_color, facecolor='none')
        ax.add_patch(rect)
        
        # Calculate font size based on ax width
        ax_width, ax_height = ax.transAxes.transform((1, 1)) - ax.transAxes.transform((0, 0))
        font_size = max(8, min(min(ax_width, ax_height) // 50, 18))  # Control range to keep size within bounds
        
        # Position the label
        if place_label_bottom:
            text_x, text_y = x1, y2
        else:
            text_x, text_y = x1, y1
        
        # Add text without background
        ax.text(text_x, text_y, label, color=edge_color, fontsize=font_size, 
                verticalalignment='top' if place_label_bottom else 'bottom')


def visualize_bbox(b_trues, b_preds, images, img_dir, vis_dir):
    os.makedirs(vis_dir, exist_ok=True)
    for image_file, ground_truth, predictions in zip(images, b_trues, b_preds):
        image = Image.open(os.path.join(img_dir, image_file))
        img_width, img_height = image.size
        fig, ax = plt.subplots(1, 1, dpi=200)
        ax.imshow(image)

        # Draw ground truth boxes
        for category, boxes in ground_truth.items():
            draw_boxes(ax, boxes, img_width, img_height, 'green', f'{category}')
        
        # Draw prediction boxes
        for category, boxes in predictions.items():
            draw_boxes(ax, boxes, img_width, img_height, 'red', f'{category}', place_label_bottom=True)
        
        # Save the result
        plt.axis('off')
        plt.savefig(os.path.join(vis_dir, image_file), bbox_inches='tight', pad_inches=0)
        plt.close()


# PanNuke, NuCLS
q_threedecimal_to_int = {
    'Please identify all nuclei in this image.': 'Please identify all nuclei in this image.',
    'Detect and classify every cell nucleus present in the picture.': 'Detect and classify every cell nucleus present in the picture.',
    'Identify all the nuclei within this image.': 'Identify all the nuclei within this image.',
    'Find and mark all nuclei in the image.': 'Find and mark all nuclei in the image.',
    'Locate every nucleus and give its category in this picture.': 'Locate every nucleus and give its category in this picture.',
    'Detect all cell nuclei in the image using bounding boxes with labels.': 'Detect all cell nuclei in the image using bounding boxes with labels.',
    'Identify every nucleus in the picture and mark them with bbox.': 'Identify every nucleus in the picture and mark them with bbox.',
    'Please use bbox to outline all nuclei and indicate every label present in this image.': 'Please use bbox to outline all nuclei and indicate every label present in this image.',
    'Distinguish all the cell nuclei in the image and use bounding boxes for each.': 'Distinguish all the cell nuclei in the image and use bounding boxes for each.',
    'Locate and mark every nucleus in the picture with a bbox.': 'Locate and mark every nucleus in the picture with a bbox.',
    'Detect and classify all nuclei in this pathology image and output with bounding boxes in [x1, y1, x2, y2] format, normalized coordinates to 0-1, accurate to three decimals.': 'Detect and classify all nuclei in this pathology image and output with bounding boxes in [x1, y1, x2, y2] format, with coordinates scaled to 0-100 as integers.',
    'Identify every cell nucleus with label in the picture, marking them with bbox in [x1, y1, x2, y2], normalize coordinates between 0 and 1, with three decimal precision.': 'Identify every cell nucleus with label in the picture, marking them with bbox in [x1, y1, x2, y2], with coordinates scaled between 0 and 100 as integers.',
    'Please use bbox to indicate all nuclei in this image, with coordinates in [x1, y1, x2, y2] format, normalized to 0-1 and rounded to three decimal places.': 'Please use bbox to indicate all nuclei in this image, with coordinates in [x1, y1, x2, y2] format, scaled to 0-100 as integers.',
    'Find all nuclei in the pathology image and represent each with a bounding box and a category, using [x1, y1, x2, y2] for normalized coordinates to a scale of 0 to 1, with three digits after the decimal.': 'Find all nuclei in the pathology image and represent each with a bounding box and a category, using [x1, y1, x2, y2] for coordinates scaled to a scale of 0 to 100 as integers.',
    'Locate and classify every nucleus in this image, using bbox for output in [x1, y1, x2, y2] format, with coordinates normalized to 0-1, and precision up to three decimals.': 'Locate and classify every nucleus in this image, using bbox for output in [x1, y1, x2, y2] format, with coordinates scaled to 0-100 as integers.',
}

# other dataset
question_list = [
    # NI_det 0730
    'Detect all nerves and classify for invasion.', 
]

def main(q_file, a_file, img_dir, vis_dir):
    qjsons, ajsons = [], []
    with open(q_file, 'r') as file:
        for line in file:
            qjsons.append(json.loads(line))
    with open(a_file, 'r') as file:
        for line in file:
            ajsons.append(json.loads(line))

    b_trues, b_preds, images, except_question_id = [], [], [], []
    for tr, pr in zip(qjsons, ajsons):
        assert tr['question_id'] == pr['question_id'], f"{tr['question_id']} vs {pr['question_id']}"
        if tr['answer'].find('<detection_result>') != -1:
            if 'int' in a_file:
                if tr['text'] in list(q_threedecimal_to_int.values()) + question_list:  # 整数
                    try:
                        # Extract each category and their bounding boxes
                        b_true = extract_bbox_categories(tr['answer'])
                        b_pred = extract_bbox_categories(pr['text'])
                        b_trues.append(b_true)
                        b_preds.append(b_pred)
                        images.append(tr['image'])
                    except:
                        print(f"================== Exception get true answer at question_id: {tr['question_id']} ==================",
                              file=sys.stderr, flush=True)
                        traceback.print_exc()
                        except_question_id.append(tr['question_id'])
            else:
                if tr['text'] in list(q_threedecimal_to_int.keys()) + question_list:  # 小数
                    try:
                        # Extract each category and their bounding boxes
                        b_true = extract_bbox_categories(tr['answer'])
                        b_pred = extract_bbox_categories(pr['text'])
                        b_trues.append(b_true)
                        b_preds.append(b_pred)
                        images.append(tr['image'])
                    except:
                        print(f"================== Exception get prediction at question_id: {tr['question_id']} ==================",
                              file=sys.stderr, flush=True)
                        traceback.print_exc()
                        except_question_id.append(tr['question_id'])

    # total:87 except_question_id:[12, 18, 61, 77, 165, 170, 179, 203, 220, 234, 242, 260, 276, 298, 305, 320, 324, 356, 381, 384, 391, 416, 425, 436, 474, 483, 530]
    assert len(b_trues) == len(b_preds) == len(images)
    print(f'total:{len(b_trues)} except_question_id:{except_question_id}')

    class_metrics, average_metrics = calculate_metrics(b_trues, b_preds)
    for cell_type, metrics in class_metrics.items():
        formatted_metrics = {key: round(value*100, 2) for key, value in metrics.items()}
        print(f'{cell_type}:', formatted_metrics)

    formatted_average_metrics = {key: round(value*100, 2) for key, value in average_metrics.items()}
    print(f'average: {formatted_average_metrics}')

    if vis_dir:
        visualize_bbox(b_trues, b_preds, images, img_dir, vis_dir)


def get_args():
    parser = argparse.ArgumentParser(description='evaluate with class detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--q_file', type=str, default="", help='path to groundtruth file')
    parser.add_argument('--a_file', type=str, default="", help='path to prediction file')
    parser.add_argument('--img_dir', type=str, default=None, help='path to image folders')
    parser.add_argument('--vis_dir', type=str, default=None, help='path to visualize folders')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args.q_file, args.a_file, args.img_dir, args.vis_dir)