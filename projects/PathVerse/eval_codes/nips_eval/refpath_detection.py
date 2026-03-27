import argparse
import os
import os.path as osp
import sys
import traceback

import cv2

from utils import load_json, load_jsonl, parse_bbox_string, compute_iou, check_negative_exist, extract_between_tags


def _extract_gt_answer(item):
    if "answer" in item:
        return item["answer"]
    if "solution" in item:
        return item["solution"]
    if "messages" in item and item["messages"]:
        for message in reversed(item["messages"]):
            if message.get("role") in ["assistant", "gpt", "bot"]:
                return message.get("content")
    return None


def _load_gt_data(gt_file):
    if gt_file is None:
        return None, None
    if gt_file.endswith(".jsonl"):
        gt_data = load_jsonl(gt_file)
    else:
        gt_data = load_json(gt_file)
    if gt_data and all("question_id" in item for item in gt_data):
        gt_by_qid = {item["question_id"]: _extract_gt_answer(item) for item in gt_data}
        return gt_data, gt_by_qid
    return gt_data, None


def _parse_boxes(text):
    text = extract_between_tags(text, "<bbox_list>", "</bbox_list>", include_tags=True, return_origin=True)
    if check_negative_exist(text):
        return []
    return parse_bbox_string(text)


def _get_image_path(resdata, gt_data, args):
    image = resdata.get("images", resdata.get("image"))
    if image is None and gt_data is not None:
        image = gt_data.get("images", gt_data.get("image"))
    if isinstance(image, (list, tuple)):
        image = image[0]
    if image and args.img_dir and not osp.exists(image) and not osp.isabs(image):
        image = osp.join(args.img_dir, image)
    return image


def _get_question(resdata, gt_data):
    question = resdata.get("prompt")
    if not question and gt_data is not None and gt_data.get("messages"):
        for message in gt_data["messages"]:
            if message.get("role") in ["user", "human", "tool"]:
                question = message.get("content")
                break
    if question:
        question = question.replace("<image>", "").strip()
    return question or ""


def get_dataset(args):
    res_dataset = load_jsonl(args.result_file)
    gt_dataset, gt_by_qid = _load_gt_data(args.gt_file)
    dataset = []
    for i, resdata in enumerate(res_dataset):
        question_id = resdata.get("question_id", i)
        gt_answer = None
        if gt_by_qid is not None:
            gt_answer = gt_by_qid.get(question_id)
        elif gt_dataset is not None:
            gt_answer = _extract_gt_answer(gt_dataset[i])
        else:
            gt_answer = resdata.get("gt_answer")
        if gt_answer is None:
            print(f"Missing gt_answer for question_id={question_id}", file=sys.stderr)
            continue
        try:
            true_boxes = _parse_boxes(gt_answer)
        except Exception:
            print(f"Error parsing ground truth: {gt_answer}", file=sys.stderr)
            traceback.print_exc()
            raise
        try:
            model_response = resdata.get("model_response", "")
            pred_boxes = _parse_boxes(model_response) if model_response else []
        except Exception:
            print(f"Error parsing prediction: {resdata.get('model_response')}", file=sys.stderr)
            traceback.print_exc()
            print("——————————————————————————————————————————————————————————————————————", file=sys.stderr)
            pred_boxes = []
        gt_data = gt_dataset[i] if gt_dataset is not None else None
        image = _get_image_path(resdata, gt_data, args)
        question = _get_question(resdata, gt_data)
        dataset.append((image, question, true_boxes, pred_boxes))
    return dataset


def evaluate_detection(data_list, iou_threshold=0.5):
    total_TP, total_FP, total_FN = 0, 0, 0
    num_negative, negative_TN, negative_FP = 0, 0, 0
    total_iou = 0

    for _, _, true_boxes, pred_boxes in data_list:
        if true_boxes:
            matched_gt, matched_pred, match_scores = set(), set(), []
            iou_pairs = []
            for i, gt in enumerate(true_boxes):
                for j, pred in enumerate(pred_boxes):
                    iou = compute_iou(gt, pred)
                    if iou >= iou_threshold:
                        iou_pairs.append((iou, i, j))
            iou_pairs.sort(reverse=True)
            for iou, i, j in iou_pairs:
                if i not in matched_gt and j not in matched_pred:
                    matched_gt.add(i)
                    matched_pred.add(j)
                    match_scores.append(iou)

            TP = len(match_scores)
            total_TP += TP
            total_FP += len(pred_boxes) - TP
            total_FN += len(true_boxes) - TP
            total_iou += sum(match_scores)
        else:
            num_negative += 1
            num_pred = len(pred_boxes)
            negative_FP += num_pred
            if num_pred == 0:
                negative_TN += 1

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    average_iou = total_iou / total_TP if total_TP > 0 else 0.0
    fpr = negative_FP / num_negative if num_negative > 0 else 0.0
    tnr = negative_TN / num_negative if num_negative > 0 else 0.0
    return {"P": precision, "R": recall, "F1": f1, "avg_IoU": average_iou, "FPR": fpr, "TNR": tnr}


def _resize_short_side(img, short_side=400):
    h, w = img.shape[:2]
    if min(h, w) == short_side:
        return img
    scale = short_side / min(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _wrap_text(text, max_width, font_face, font_scale, thickness):
    if not text:
        return [""]
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        text_size = cv2.getTextSize(candidate, font_face, font_scale, thickness)[0]
        if text_size[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def visualize_bbox(vis_dir, data_list, short_side=400):
    os.makedirs(vis_dir, exist_ok=True)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_margin = 6
    for image, question, true_boxes, pred_boxes in data_list:
        if not image or not osp.exists(image):
            print(f"Image not found: {image}", file=sys.stderr)
            continue
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to load image: {image}", file=sys.stderr)
            continue
        img = _resize_short_side(img, short_side=short_side)

        max_text_width = max(10, img.shape[1] - text_margin * 2)
        lines = _wrap_text(question, max_text_width, font_face, font_scale, thickness)
        line_height = cv2.getTextSize("Ag", font_face, font_scale, thickness)[0][1]
        text_block_height = (line_height + 4) * len(lines) + text_margin * 2
        canvas = img
        if text_block_height > 0:
            canvas = cv2.copyMakeBorder(
                img,
                top=text_block_height,
                bottom=0,
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            y = text_margin + line_height
            for line in lines:
                cv2.putText(canvas, line, (text_margin, y), font_face, font_scale, (20, 20, 20), thickness)
                y += line_height + 4

        for box in true_boxes:
            x1, y1, x2, y2 = [round(coord / 1000 * (canvas.shape[1] if i % 2 == 0 else canvas.shape[0] - text_block_height))
                              for i, coord in enumerate(box)]
            y1 += text_block_height
            y2 += text_block_height
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 192, 0), max(1, int(min(canvas.shape[0], canvas.shape[1]) * 0.002)))

        for box in pred_boxes:
            x1, y1, x2, y2 = [round(coord / 1000 * (canvas.shape[1] if i % 2 == 0 else canvas.shape[0] - text_block_height))
                              for i, coord in enumerate(box)]
            y1 += text_block_height
            y2 += text_block_height
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 0, 192), max(1, int(min(canvas.shape[0], canvas.shape[1]) * 0.002)))

        out_name = osp.splitext(osp.basename(image))[0] + ".jpg"
        cv2.imwrite(osp.join(vis_dir, out_name), canvas)


def main(args):
    dataset = get_dataset(args)
    if not dataset:
        print("No samples to evaluate.")
        return
    res = evaluate_detection(dataset, iou_threshold=args.iou_threshold)
    print(f"Samples: {len(dataset)}")
    print(f"Precision: {res['P']*100:.2f}, Recall: {res['R']*100:.2f}, F1-score: {res['F1']*100:.2f}, Average IoU: {res['avg_IoU']*100:.2f}")
    print(f"False Positive Rate: {res['FPR']*100:.2f}, True Negative Rate: {res['TNR']*100:.2f}")
    if args.vis_dir:
        visualize_bbox(args.vis_dir, dataset, short_side=400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", required=True)
    parser.add_argument("--gt_file", default=None)
    parser.add_argument("--img_dir", default=None)
    parser.add_argument("--vis_dir", default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
