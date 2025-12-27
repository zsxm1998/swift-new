import os
import os.path as osp
import sys
import fnmatch
import cv2
import argparse
import traceback
from utils import load_json, load_jsonl, check_negative_exist, parse_bbox_string, compute_iou, extract_between_tags


question_lists = {
    'nucleus': [
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
        'Detect all nuclei in this pathology image and output with bounding boxes in [x1, y1, x2, y2] format, with coordinates scaled to 0-1000 as integers.',
        'Identify every cell nucleus in the picture, marking them with bbox in [x1, y1, x2, y2], with coordinates rescaled between 0 and 1000 as integers.',
        'Please use bbox to indicate all nuclei in this image, with coordinates in [x1, y1, x2, y2] format, scaled to 0-1000 as integers.',
        'Find all nuclei in the pathology image and represent each with a bounding box, using [x1, y1, x2, y2] for coordinates rescaled to a scale of 0 to 1000 as integers.',
        'Locate every nucleus in this image, using bbox for output in [x1, y1, x2, y2] format, with coordinates scaled to 0-1000 as integers.',
    ],
    'vessel': [
        "Detect all vessels",
        "Find every blood vessel.",
        "Identify all vessels in image",
        "Locate all blood vessels.",
        "Can you detect all blood vessels in this image?",
        "Could you show all the vessels in the image?",
        "Locate and mark every blood vessel in this picture?",
        "Please identify and create bounding boxes around every blood vessel visible in this image, including both large and small vessels.",
    ],
    'cancerous_nuclei_in_mvi_vessel': [
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
    ],
    'lymph_node': [
        "Detect all lymph nodes.",
    ],
    'nerve': [
        "Detect all nerves.",
    ]
}


def get_dataset(args, parse_func=parse_bbox_string, stag='<bbox_list>', etag='</bbox_list>', null_func=list, valid_questions=None):
    dataset, gt_dataset = [], None
    res_dataset = load_jsonl(args.result_file)
    if args.gt_file is not None:
        gt_dataset = load_json(args.gt_file) if args.gt_file.endswith('.json') else load_json(args.gt_file)
        assert len(gt_dataset) == len(res_dataset), f'{len(gt_dataset)=} but {len(res_dataset)}'
    for i, resdata in enumerate(res_dataset):
        question = resdata['prompt'].replace('<image>','').strip()
        if valid_questions is not None and question not in valid_questions:
            continue
        if gt_dataset is not None:
            gtdata = gt_dataset[i]
            assert gtdata.get('question_id', i) == resdata.get('question_id', i), f'{gtdata.get("question_id", i)=} but {resdata.get("question_id", i)=}'
            image = gtdata.get('images', gtdata.get('image'))
            if isinstance(image, (list, tuple)):
                image = image[0]
            if args.img_dir and not osp.exists(image) and not osp.isabs(image):
                image = osp.join(args.img_dir, image)
            if args.dataset and not fnmatch.fnmatch(image, args.dataset):
                continue
        else:
            image = resdata.get('images', resdata.get('image'))
            if isinstance(image, (list, tuple)):
                image = image[0]
            if image is not None and args.img_dir and not osp.exists(image) and not osp.isabs(image):
                image = osp.join(args.img_dir, image)
            if image is not None and args.dataset and not fnmatch.fnmatch(image, args.dataset):
                continue
        gt_answer = extract_between_tags(resdata['gt_answer'], stag, etag, include_tags=True, return_origin=True)
        model_response = extract_between_tags(resdata['model_response'], stag, etag, include_tags=True, return_origin=True)
        if check_negative_exist(gt_answer):
            true_boxes = null_func()
        else:
            try:
                true_boxes = parse_func(gt_answer)
            except:
                print(f"Error parsing ground truth: {resdata['gt_answer']}", file=sys.stderr)
                raise
        try:
            if check_negative_exist(model_response):
                pred_boxes = null_func()
            else:
                pred_boxes = parse_func(model_response)
        except:
            print(f"Error parsing prediction: {resdata['model_response']}", file=sys.stderr)
            traceback.print_exc()
            print('——————————————————————————————————————————————————————————————————————', file=sys.stderr)
            pred_boxes = null_func()
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


def visualize_bbox(vis_dir, data_list):
    os.makedirs(vis_dir, exist_ok=True)
    for image, _, true_boxes, pred_boxes in data_list:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        # Draw each true bounding box in green
        for box in true_boxes:
            if type(box[0]) == int:
                x1, y1, x2, y2 = [round(coord / 1000 * (img.shape[1] if i % 2 == 0 else img.shape[0]))
                                  for i, coord in enumerate(box)]
            else:
                x1, y1, x2, y2 = [round(coord * (img.shape[1] if i % 2 == 0 else img.shape[0]))
                                  for i, coord in enumerate(box)]
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 192, 0), max(1, int(min(img.shape[0], img.shape[1])*0.002)))
        # Draw each pred bounding box in red
        for box in pred_boxes:
            if type(box[0]) == int:
                x1, y1, x2, y2 = [round(coord / 1000 * (img.shape[1] if i % 2 == 0 else img.shape[0]))
                                  for i, coord in enumerate(box)]
            else:
                x1, y1, x2, y2 = [round(coord * (img.shape[1] if i % 2 == 0 else img.shape[0]))
                                  for i, coord in enumerate(box)]
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 0, 192), max(1, int(min(img.shape[0], img.shape[1])*0.002)))
        # Save image
        cv2.imwrite(osp.join(vis_dir, osp.splitext(osp.basename(image))[0]+'.jpg'), img)


def main(args):
    all_questions = {question for sublist in question_lists.values() for question in sublist}
    dataset = get_dataset(args, valid_questions=all_questions)

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
        res = evaluate_detection(data_list)
        print(f'Precision: {res["P"]*100:.2f}, Recall: {res["R"]*100:.2f}, F1-score: {res["F1"]*100:.2f}, Average IoU: {res["avg_IoU"]*100:.2f}')
        print(f'False Positive Rate: {res["FPR"]*100:.2f}, True Negative Rate: {res["TNR"]*100:.2f}', end='\n\n')

        if args.vis_dir:
            vis_dir = args.vis_dir if only_one_data_type else osp.join(args.vis_dir, data_type)
            visualize_bbox(vis_dir, data_list)


if __name__ == '__main__':
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