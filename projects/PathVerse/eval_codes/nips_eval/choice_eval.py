import re
import fnmatch
import argparse
import warnings
from sklearn import metrics
from utils import load_json, load_jsonl, extract_choice_label, extract_between_tags

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


# 同一个问题选项多的放前面，防止误匹配
question_lists = {
    'pathological_type': [
        # XiaJiabin
        ("What is the pathological type? A. {}, B. {}, C. {}, D. {}, E. {}, F. {}, G. {}", ['A', 'B', 'C', 'D', 'E', 'F', 'G']),
        # liver_subtype_patho_thumb
        ("What is the pathological type? A. {}, B. {}, C. {}", ['A', 'B', 'C']),
        ("Diagnose the disease from this image. A. {}, B. {}, C. {}, D. {}", ['A', 'B', 'C', 'D']),
        # liver_subtype_patho
        ("What is the pathological type? A. {}, B. {}", ['A', 'B']),
        ("Diagnose the disease from this image. A. {}, B. {}, C. {}", ['A', 'B', 'C']),
    ],
    'subtype_cancer': [
        ("Identify the cancer subtype. A. {}, B. {}, C. {}, D. {}", ['A', 'B', 'C', 'D']),
        ("Identify the cancer subtype. A. {}, B. {}, C. {}", ['A', 'B', 'C']),
        ("Identify the cancer subtype. A. {}, B. {}", ['A', 'B']),
        ("Identify the cancer subtype from the options: A. {}, B. {}", ['A', 'B']),
    ],
    'LAUREN_type': [
        ("Identify the lauren classification from the options: A. {}, B. {}, C. {}", ['A', 'B', 'C']),
        ("Identify the lauren classification from the options: A. {}, B. {}", ['A', 'B']),
    ],
    'cancer_grading': [
        ("What is the cancer grade? A. {}, B. {}, C. {}, D. {}, E. {}", ['A', 'B', 'C', 'D', 'E']),
        ("What is the cancer grade? A. {}, B. {}, C. {}", ['A', 'B', 'C']),
        ("Grade the cancer in this image A. {}, B. {}, C. {}", ['A', 'B', 'C']), # for ShanghaiBreast
    ],
}


def parse_question_options(question_tuple, filled_question):
    question_template, options_format = question_tuple
    # 将每个选项的占位符 {} 替换为贪婪匹配的正则表达式模式
    pattern = re.escape(question_template).replace(r'\{\}', r'(.+)')
    # 尝试使用正则表达式匹配实际填充的问题字符串
    match = re.match(pattern, filled_question)
    filled_options = None
    if match:
        # 提取匹配的选项内容
        filled_options = list(match.groups())
        if len(filled_options) != len(options_format) or any([not x for x in filled_options]):
            filled_options = None
    return filled_options


def get_dataset(args):
    dataset, gt_dataset = [], None
    res_dataset = load_jsonl(args.result_file)
    if args.gt_file is not None:
        gt_dataset = load_json(args.gt_file) if args.gt_file.endswith('.json') else load_json(args.gt_file)
        assert len(gt_dataset) == len(res_dataset), f'{len(gt_dataset)=} but {len(res_dataset)}'
    for i, resdata in enumerate(res_dataset):
        if gt_dataset is not None:
            gtdata = gt_dataset[i]
            assert gtdata.get('question_id', i) == resdata.get('question_id', i), f'{gtdata.get("question_id", i)=} but {resdata.get("question_id", i)=}'
            image = gtdata.get('images', gtdata.get('image'))
            if isinstance(image, (list, tuple)):
                image = image[0]
            if args.dataset and not fnmatch.fnmatch(image, args.dataset):
                continue
        else:
            image = resdata.get('images', resdata.get('image'))
            if isinstance(image, (list, tuple)):
                image = image[0]
            if image is not None and args.dataset and not fnmatch.fnmatch(image, args.dataset):
                continue
        question = resdata['prompt'].replace('<image>','').strip()
        dataset.append((question, resdata['gt_answer'], resdata['model_response']))
    return dataset


def main(args):
    dataset = get_dataset(args)
    results_dict = {key:{'names':[], 'gt':[], 'pr':[]} for key in question_lists}
    total_count, correct_count = 0, 0
    unknown_questions = set()
    for question, gt, pred in dataset:
        for data_type, format_list in question_lists.items():
            for question_tuple in format_list:
                options = parse_question_options(question_tuple, question)
                if options:
                    options = [opt.strip().lower() for opt in options]
                    for opt in sorted(options):
                        if opt not in results_dict[data_type]['names']:
                            results_dict[data_type]['names'].append(opt)
                    gtlabel = extract_choice_label(gt)
                    if not gtlabel:
                        gtindex = -1
                        results_dict[data_type]['gt'].append(-1)
                    else:
                        gtindex = question_tuple[1].index(gtlabel)
                        results_dict[data_type]['gt'].append(results_dict[data_type]['names'].index(options[gtindex]))
                    total_count += 1

                    pred = extract_between_tags(pred, '<answer>', '</answer>', return_origin=True)
                    try:
                        if not (prlabel := extract_choice_label(pred)):
                            raise ValueError(pred)
                        prindex = question_tuple[1].index(prlabel)
                    except:
                        prindex = -1
                        results_dict[data_type]['pr'].append(-1)
                    else:
                        results_dict[data_type]['pr'].append(results_dict[data_type]['names'].index(options[prindex]))
                    if gtindex == prindex:
                        correct_count += 1
                    break
            else:
                continue
            break
        else:
            # 单独处理未见问题
            if args.report_unknown_question and question not in unknown_questions:
                print(f'Unknown question: {question}')
                unknown_questions.add(question)
            if 'Unknown Questions' not in results_dict:
                results_dict['Unknown Questions'] = {'names': None, 'gt': [], 'pr': []}
            gtlabel = extract_choice_label(gt) or 'Not_Chosen'
            results_dict['Unknown Questions']['gt'].append(gtlabel)
            pred = extract_between_tags(pred, '<answer>', '</answer>', return_origin=True)
            prlabel = extract_choice_label(pred) or 'Not_Chosen'
            results_dict['Unknown Questions']['pr'].append(prlabel)
            total_count += 1
            if gtlabel == prlabel:
                correct_count += 1

    print(f'Overall accuracy: {correct_count/total_count:.2%} ({correct_count}/{total_count})')
    for key, value in results_dict.items():
        if not value['gt']:
            continue
        print(f'---------------------[{key}]---------------------')
        print(f'Accuracy: {metrics.accuracy_score(value["gt"], value["pr"]):.2%}',
              f'Balanced Accuracy: {metrics.balanced_accuracy_score(value["gt"], value["pr"]):.2%}')
        labels = None if value['names'] is None else list(range(len(value['names'])))
        print(metrics.classification_report(value['gt'], value['pr'], digits=4, zero_division=0, labels=labels, target_names=value['names']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation for choice question', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_file', type=str, required=True, help='path to the generation result jsonl file')
    parser.add_argument('--gt_file', type=str, default=None, help='path to groundtruth file')
    parser.add_argument('--dataset', type=str, default=None, help='only evaluate images of which path contains the dataset name')
    parser.add_argument('--report_unknown_question', action='store_true', help='report unknown question')
    args = parser.parse_args()
    if args.dataset is not None: # for fnmatch
        args.dataset = '*' + args.dataset + '*'
    main(args)
