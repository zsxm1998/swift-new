import argparse
import os
import json
from sklearn import metrics

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for CVPR MVI test set', add_help=False)
    parser.add_argument('--gt', type=str, default="test.json", help='path to groundtruth file', )
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file', )
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def evaluate(gt, pred):
    true_data = load_jsonl(gt)
    pred_data = load_jsonl(pred)

    true_list, pred_list = [], []
    for td, pd in zip(true_data, pred_data):
        assert td['question_id'] == pd['question_id'], f"{td['question_id']} != {pd['question_id']}"
        if td['answer'].startswith('Yes'):
            true_list.append(1)
        elif td['answer'].startswith('No'):
            true_list.append(0)
        else:
            raise ValueError(td['answer'])
        if pd['text'].startswith('Yes'):
            pred_list.append(1)
        elif pd['text'].startswith('No'):
            pred_list.append(0)
        else:
            raise ValueError(pd['text'])
        
    print(metrics.classification_report(true_list, pred_list, digits=4, zero_division=0))



if __name__ == '__main__':
    args = parse_option()
    # perform evaluation
    evaluate(args.gt, args.pred)
