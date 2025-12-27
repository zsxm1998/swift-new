import argparse
import os
import json
from sklearn import metrics

cancer_type_questions = [
    'Diagnose the disease from this image.',
    "Analyze this image to determine the patient's disease.",
    "Use this image to diagnose the patient's illness.",
    "What disease could this pathology slide be from?",
    'Identify cancer type',
    "What is the cancer type in this image?",
    "Identify the cancer type in this pathology image.",
    "Can you determine the cancer type shown in this image?",
    "What category does this cancer fall under?",
    "Classify the cancer type depicted in this pathology image.",
    "Could you identify the specific type of cancer in this image?",
    "What is the classification of the cancer in this pathology slide?",
    "Based on the image, what is the cancer type and classification?",
    "Please provide the cancer type and classification for this pathology image.",
    "Analyze the pathology image and determine the exact cancer type and classification.",
]
subtype_cancer = [
    "Identify the cancer subtype.",
    "What is the cancer subtype?",
]
grade_cancer = ["Grade the cancer in this image"]
patho_type = [
    "What is the pathological type?",
    "Diagnose the disease from this image.",
]
LNM_cls = ["Is this lymph node metastasis?",]
NI_cls = ["Is this neural invasion?",]
GBM_MGMT = ["Can you detect MGMT methylation from this slide?",]
LAUREN_type = [
    "Identify the lauren classification.",
    "What is the lauren classification?",
]
question_lists = {
    'TCGA_Uniform_Tumor': cancer_type_questions,
    'liver_subtype_patho': patho_type,
    'patho_type': patho_type,
    'LNM': LNM_cls,
    'NI': NI_cls,
    'subtype_cancer': subtype_cancer,
    'grade_cancer': grade_cancer,
    'GBM_MGMT': GBM_MGMT,
    'LAUREN_type': LAUREN_type,
}

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for multi type classification', add_help=False)
    parser.add_argument('--gt', type=str, default="test.jsonl", help='path to groundtruth file', )
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file', )
    parser.add_argument('--dataset', type=str, default=None, help='key to find question list', )
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def evaluate(gt, pred, dataset):
    true_data = load_jsonl(gt)
    pred_data = load_jsonl(pred)

    true_list, pred_list = [], []
    yesnoflag = False
    for td, pd in zip(true_data, pred_data):
        assert td['question_id'] == pd['question_id'], f"{td['question_id']} != {pd['question_id']}"
        if dataset is None or td['text'] in question_lists[dataset]:
            # 这里加上开头不为normal是为RCC数据集而改变的。
            if td['answer'].lower().startswith('yes') or td['answer'].lower().startswith('no') and not td['answer'].lower().startswith('normal'):
                yesnoflag = True
                if td['answer'].lower().startswith('yes'):
                    true_list.append(1)
                elif td['answer'].lower().startswith('no'):
                    true_list.append(0)
                if pd['text'].lower().startswith('yes'):
                    pred_list.append(1)
                elif pd['text'].lower().startswith('no'):
                    pred_list.append(0)
            else:
                true_list.append(td['answer'].replace('.','').lower())
                pred_list.append(pd['text'].replace('.','').lower())
    
    assert len(true_list) == len(pred_list), f'{len(true_list)=} != {len(pred_list)=}'
    # all_labels = sorted(set(true_list) | set(pred_list))
    all_labels = sorted(set(true_list))
    print(f'True category number: {len(set(true_list))}, Pred category number: {len(set(pred_list))}')
    if yesnoflag:
        print(metrics.classification_report(true_list, pred_list, digits=4, zero_division=0))
    else:
        print(metrics.classification_report(true_list, pred_list, digits=4, zero_division=0, labels=all_labels, target_names=all_labels))
    print('Accuracy:', round(metrics.accuracy_score(true_list, pred_list), 4))
    print('Balanced ACC:', round(metrics.balanced_accuracy_score(true_list, pred_list), 4))


if __name__ == '__main__':
    args = parse_option()
    # perform evaluation
    evaluate(args.gt, args.pred, args.dataset)
