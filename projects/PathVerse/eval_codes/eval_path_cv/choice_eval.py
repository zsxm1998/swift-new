import argparse
import re
import json
from sklearn import metrics


pathological_type = [
    # liver_subtype_patho
    ("What is the pathological type? A. {}, B. {}", ['A', 'B']),
    ("Diagnose the disease from this image. A. {}, B. {}, C. {}", ['A', 'B', 'C']),
    # liver_subtype_patho_thumb
    ("What is the pathological type? A. {}, B. {}, C. {}", ['A', 'B', 'C']),
    ("Diagnose the disease from this image. A. {}, B. {}, C. {}, D. {}", ['A', 'B', 'C', 'D']),
    # XiaJiabin
    ("What is the pathological type? A. {}, B. {}, C. {}, D. {}, E. {}, F. {}, G. {}", ['A', 'B', 'C', 'D', 'E', 'F', 'G']),
]
cancer_grading = [
    ("What is the cancer grade? A. {}, B. {}, C. {}", ['A', 'B', 'C']),
    ("What is the cancer grade? A. {}, B. {}, C. {}, D. {}, E. {}", ['A', 'B', 'C', 'D', 'E']),
]
liver_subtype_HCC_subtype = [("Identify the cancer subtype. A. {}, B. {}, C. {}, D. {}", ['A', 'B', 'C', 'D']),]
subtype_cancer = [
    ("Identify the cancer subtype. A. {}, B. {}", ['A', 'B']),
    ("Identify the cancer subtype. A. {}, B. {}, C. {}", ['A', 'B', 'C']),
    ("Identify the cancer subtype. A. {}, B. {}, C. {}, D. {}", ['A', 'B', 'C', 'D']),
    ("Identify the cancer subtype from the options: A. {}, B. {}", ['A', 'B']),
]
prognosis_questions = [("What is the prognosis for this patient? A. {}, B. {}", ['A', 'B']),]
LAUREN_type = [
    ("Identify the lauren classification from the options: A. {}, B. {}", ['A', 'B']),
    ("Identify the lauren classification from the options: A. {}, B. {}, C. {}", ['A', 'B', 'C']),
]
question_lists = {
    'liver_subtype_patho': pathological_type[0:2],
    'prognosis': prognosis_questions,
    'HCC_patch_grading': cancer_grading[1:2],
    'liver_subtype_ICC_grading': cancer_grading[0:1],
    'liver_subtype_HCC_subtype': liver_subtype_HCC_subtype,
    'liver_subtype_patho_thumb': pathological_type[2:4],
    'ICC_subtype': subtype_cancer[0:1],
    'RCC_patho': pathological_type[0:1],
    'RCC_subtype': subtype_cancer[2:3],
    'XiaJiabin': pathological_type[4:5],
    'five_class_grading': cancer_grading[1:2],
    'Lung_1000_subtype': subtype_cancer[3:4],
    'LAUREN_type_two': LAUREN_type[0:1],
    'LAUREN_type_three': LAUREN_type[1:2],
}


def parse_question_and_answer(question_tuple, filled_question, answer=None):
    question_template, options_format = question_tuple
    
    # 将每个选项的占位符 {} 替换为贪婪匹配的正则表达式模式
    pattern = re.escape(question_template).replace(r'\{\}', r'(.+)')
    
    # 尝试使用正则表达式匹配实际填充的问题字符串
    match = re.match(pattern, filled_question)
    
    if match:
        # 提取匹配的选项内容
        filled_options = list(match.groups())
        assert len(filled_options) == len(options_format)
        filled_answer = None
        if answer:
            candidates = [x.format(y) for x, y in zip(options_format, filled_options)]
            # 在 filled_options 中逐一匹配 answer，找出实际答案
            for format_option, actual_option in zip(options_format, filled_options):
                full_option = format_option.format(actual_option)
                if answer == full_option:
                    filled_answer = actual_option
                    break
            else:
                for format_option, actual_option in zip(options_format, filled_options):
                    index = min(len(format_option) if format_option.find('{')==-1 else format_option.find('{'), len(answer))
                    if answer[:index] == format_option[:index]:
                        filled_answer = actual_option
                        break
                else:
                    raise ValueError(f"Can't find answer={answer}")
        
        return filled_options, filled_answer
    else:
        return None, None

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for choice question', add_help=False)
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
    if dataset is None:
        true_list, pred_list = [], []
        for td, pd in zip(true_data, pred_data):
            assert td['question_id'] == pd['question_id'], f"{td['question_id']} != {pd['question_id']}"
            true_list.append(td['answer'].lower()[0])
            pred_list.append(pd['text'].lower()[0])
        assert len(true_list) == len(pred_list)
        all_labels = sorted(set(true_list))
        print(f'True category number: {len(set(true_list))}, Pred category number: {len(set(pred_list))}')
        print(metrics.classification_report(true_list, pred_list, digits=4, zero_division=0, labels=all_labels, target_names=all_labels))
        print('Accuracy:', round(metrics.accuracy_score(true_list, pred_list), 4))
        print('Balanced ACC:', round(metrics.balanced_accuracy_score(true_list, pred_list), 4))
    else:
        options = set()
        for td in true_data:
            for x in question_lists[dataset]:
                ol = parse_question_and_answer(x, td['text'])[0]
                if ol:
                    ol = [x.lower() for x in ol]
                    options.update(ol)
                    break
        options = sorted(list(options))

        true_list, pred_list = [], []
        for td, pd in zip(true_data, pred_data):
            assert td['question_id'] == pd['question_id'], f"{td['question_id']} != {pd['question_id']}"
            for x in question_lists[dataset]:
                ol, ta = parse_question_and_answer(x, td['text'], td['answer'])
                if ol:
                    true_list.append(options.index(ta.lower()))
                    _, pa = parse_question_and_answer(x, td['text'], pd['text'])
                    pred_list.append(options.index(pa.lower()))
                    break
        assert len(true_list) == len(pred_list)
        print(metrics.classification_report(true_list, pred_list, digits=4, zero_division=0, labels=range(len(options)), target_names=options))
        print('Accuracy:', round(metrics.accuracy_score(true_list, pred_list), 4))
        print('Balanced ACC:', round(metrics.balanced_accuracy_score(true_list, pred_list), 4))


if __name__ == '__main__':
    args = parse_option()
    evaluate(args.gt, args.pred, args.dataset)
