import os
import os.path as osp
import cv2
import numpy as np
import argparse

from no_class_detection import get_dataset
from utils import parse_polygon_string


question_lists = {
    'cancer_area': [
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
        'Does this pathology image contain cancer? If yes, provide the boundaries of each cancer area as polygons, with vertex coordinates (x, y) rescaled between 0 and 1000 as integers.',
        'Is there cancer in this pathology picture? If present, outline each cancer region with a polygon, using rescaled coordinates [x, y] up to 1000 as integers.',
        'Can you detect cancer areas in this pathology image? For each, output a boundary polygon with vertices (x, y) scaled to 0-1000 as integers.',
        'Check this pathology image for cancer regions. If found, draw polygons around each area, with vertex coordinates [x, y] rescaled and rounded to 1000 as integers.',
        'Are there any cancerous areas in this pathology image? If so, illustrate their boundaries as polygons, with (x, y) vertices scaled to a range of 0 to 1000 as integers.',
    ],
    'diagnostic_area': [ # ZheYi0607/liver_cancer/thumbnails
        "Segment the diagnostic area.",
        "Can you segment the key area?",
        "Please identify and segment the diagnostic area in this image.",
        "Could you mark and segment the key area in this pathology image?",
        "Can you detect and segment the diagnostic areas relevant to the cancer diagnosis?",
        "Please analyze the image and provide a detailed segmentation of the key diagnostic areas.",
    ],
    'nerve': [ # ZheYi0730/NI_det and ZheYi0730/NI_cls 分割健康神经
        "Segment all nerves.",
        "Can you segment the nerves in this image?",
        "Identify and segment all nerves present in the pathology image.",
        "Please detect and segment all nerves in this pathology slide.",
        "Could you locate, identify, and segment every nerve visible in this pathology image?",
    ], 
    'cancer_in_lymph_node': [ #LNM
        "Segment the cancerous area in the lymph node.",
        "Can you segment the cancerous region in this lymph node?",
        "Please identify and segment the cancerous areas within this lymph node.",
        "Could you analyze and segment all the cancerous regions in the lymph node shown in this image?",
    ],
    'cancer_in_nerve': [ #NI_cls
        "Segment the cancerous area in the nerve.",
        "Can you segment the cancerous region in this nerve?",
        "Please identify and segment the cancerous areas within this nerve.",
        "Could you analyze and segment all the cancerous regions in the nerve shown in this image?",
        "Can you detect and segment the specific areas of cancer within the nerve in this pathology image?",
    ]
}


def contours_to_mask(contours, shape=1000): # shape=(H, W)
    if isinstance(shape, int):
        shape = (shape, shape)
    mask = np.zeros(shape, dtype=np.uint8)
    new_contours = []
    for contour in contours:
        if len(contour) == 0:
            continue
        contour = np.array(contour) / 1000 * np.array([shape[1], shape[0]])
        contour = np.round(contour).astype(int)
        new_contours.append(contour)
    if new_contours:
        cv2.fillPoly(mask, new_contours, color=1)
    return mask


def evaluate_segmentation(data_list):
    total_TP, total_FP, total_FN = 0, 0, 0

    for _, _, true_contours, pred_contours in data_list:
        gt_mask = contours_to_mask(true_contours)
        pred_mask = contours_to_mask(pred_contours)

        TP = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
        FP = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
        FN = np.logical_and(pred_mask == 0, gt_mask == 1).sum()

        total_TP += TP
        total_FP += FP
        total_FN += FN

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    dice = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0 # Dice和F1相同
    iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0.0

    return {"P": precision, "R": recall, "Dice": dice, "IoU": iou} # F1就是Dice


def visualize_contour(vis_dir, data_list):
    os.makedirs(vis_dir, exist_ok=True)
    for image, _, true_contours, pred_contours in data_list:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        # Draw each true contour in green
        for polygon in true_contours:
            if type(polygon[0][0]) == int:
                poly_points = np.array([[[int(p[0] / 1000 * img.shape[1]), int(p[1] / 1000 * img.shape[0])]]
                                        for p in polygon], np.int32)
            else:
                poly_points = np.array([[[int(p[0] * img.shape[1]), int(p[1] * img.shape[0])]]
                                        for p in polygon], np.int32)
            cv2.polylines(img, [poly_points], True, (0, 255, 0), max(1, int(min(img.shape[0], img.shape[1])*0.004)))
        # Draw each pred contour in red
        for polygon in pred_contours:
            if type(polygon[0][0]) == int:
                poly_points = np.array([[[int(p[0] / 1000 * img.shape[1]), int(p[1] / 1000 * img.shape[0])]]
                                        for p in polygon], np.int32)
            else:
                poly_points = np.array([[[int(p[0] * img.shape[1]), int(p[1] * img.shape[0])]]
                                        for p in polygon], np.int32)
            cv2.polylines(img, [poly_points], True, (0, 0, 255), max(1, int(min(img.shape[0], img.shape[1])*0.004)))
        # Save image
        cv2.imwrite(osp.join(vis_dir, osp.splitext(osp.basename(image))[0]+'.jpg'), img)


def main(args):
    all_questions = {question for sublist in question_lists.values() for question in sublist}
    dataset = get_dataset(args, parse_func=parse_polygon_string, stag='<contour_list>',
                          etag='</contour_list>', valid_questions=all_questions)

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
        res = evaluate_segmentation(data_list)
        print(f'Precision: {res["P"]*100:.2f}, Recall: {res["R"]*100:.2f}, Dice (F1-score): {res["Dice"]*100:.2f}, IoU: {res["IoU"]*100:.2f}', end='\n\n')

        if args.vis_dir:
            vis_dir = args.vis_dir if only_one_data_type else osp.join(args.vis_dir, data_type)
            visualize_contour(vis_dir, data_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate no class segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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