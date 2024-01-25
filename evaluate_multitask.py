import os
import re
import json
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_dataset(path_to_data_file: str) -> dict:
    with open(path_to_data_file, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset


def save_data(path_to_data_file: str, data):
    with open(path_to_data_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def get_pred_value(prediction):
    if type(prediction) == str:
        option = re.search(r'\W\w\W', prediction)
        if option and len(prediction) == 3:
            mapping = {'(a)': 'change high', '(b)': 'change medium', '(c)': 'change low',
                       '(d)': 'neutral high', '(e)': 'neutral medium', '(f)': 'neutral low',
                       '(g)': 'sustain high', '(h)': 'sustain medium', '(i)': 'sustain low'}
            prediction = mapping[prediction.lower().strip()]
        return prediction.lower().strip()

    elif type(prediction) == dict:
        targets = []
        if 'Motivational level' in prediction:
            targets.append(prediction['Motivational level'])
        if 'Certainty level' in prediction:
            targets.append(prediction['Certainty level'])
        if 'Answer' in prediction:
            targets.append(prediction['Answer'])
        return ' '.join(targets).lower().strip()


def get_mismatch_output(golds: list, preds: list):
    num = 0
    labels = []
    for pred in preds:
        if pred not in golds:
            num += 1
            labels.append(pred)
    return num, labels


def calculate_metrics(golds: list, predictions: list) -> dict:
    gold_labels = list(set(golds))
    pred_labels = list(set(predictions))

    accuracy = accuracy_score(y_true=golds, y_pred=predictions)
    report = classification_report(y_true=golds, y_pred=predictions, output_dict=True)
    report.update({"accuracy": accuracy})

    confusion = confusion_matrix(y_true=golds, y_pred=predictions, labels=gold_labels)
    report.update({"confusion_max": gold_labels + confusion.tolist()})

    if len(gold_labels) != len(pred_labels):
        if not set(pred_labels).issubset(set(gold_labels)):
            num_mismatch, label_mismatch = get_mismatch_output(golds, predictions)
            report.update({"Number of mismatch output": num_mismatch,
                           "Label mismatch": list(set(label_mismatch))})
            print("Mismatch labels.")
    return report


def evaluate():
    path_to_data_file = f"{args.path_to_pred_dir}/raw_outputs/{args.input_file}"
    path_to_eval_file = f"{args.path_to_eval_dir}/[mismatch]_{args.input_file}" if args.allow_mismatch \
        else f"{args.path_to_eval_dir}/{args.input_file}"
    assert os.path.exists(path_to_data_file), "File Not Exist"

    data = load_dataset(path_to_data_file)
    report = {}

    mismatch = []
    golds, predictions = [], []
    golds_att, preds_att, golds_cer, preds_cer = [], [], [], []
    for utt_id, utt_info in data.items():
        gold = utt_info['target'].lower().strip()
        pred = get_pred_value(utt_info['prediction']).lower().strip()

        golds.append(gold)  # multitask cal
        golds_att.append(gold.split(' ')[0])
        golds_cer.append(gold.split(' ')[1])

        pred_att = re.findall(r'change|neutral|sustain', pred)
        pred_cer = re.findall(r'high|medium|low', pred)

        preds_att.append(pred_att[0]) if len(pred_att) > 0 else mismatch.append(pred)
        preds_cer.append(pred_cer[0]) if len(pred_cer) > 0 else mismatch.append(pred)
        predictions.append(f"{pred_att[0]} {pred_cer[0]}") if len(pred_att) > 0 and len(pred_cer) > 0 \
            else mismatch.append(pred)

        if len(pred_att) == 0:
            preds_att.append(pred) if args.allow_mismatch else preds_att.append('neutral')
        if len(pred_cer) == 0:
            preds_cer.append(pred) if args.allow_mismatch else preds_cer.append('medium')
        if len(pred_att) == 0 or len(pred_cer) == 0:
            predictions.append(pred) if args.allow_mismatch else predictions.append('neutral medium')

    if len(mismatch) > 0:
        print(f"Mismatch in generation:", mismatch)

    report['attitude'] = calculate_metrics(golds=golds_att, predictions=preds_att)
    report['certainty'] = calculate_metrics(golds=golds_cer, predictions=preds_cer)
    report['multitask'] = calculate_metrics(golds=golds, predictions=predictions)
    report['mismatch'] = mismatch
    save_data(path_to_data_file=path_to_eval_file, data=report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_pred_dir', type=str, default=None)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--allow_mismatch', default=False)
    args = parser.parse_args()

    args.path_to_eval_dir = f"{args.path_to_pred_dir}/evaluated"
    if not os.path.exists(args.path_to_eval_dir):
        os.makedirs(args.path_to_eval_dir, exist_ok=True)

    evaluate()
