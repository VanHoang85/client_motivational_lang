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
        pred_att = re.findall(r'change|neutral|sustain', prediction)
        pred_cer = re.findall(r'high|medium|low', prediction)
        if pred_att:
            return pred_att[0]
        if pred_cer:
            return pred_cer[0]

        if args.allow_mismatch:
            return 'neutral' if task == 'attitude' else 'medium'
        return prediction.lower().strip()

    elif type(prediction) == dict:
        targets = []
        if 'Motivational level' in prediction:
            targets.append(prediction['Motivational level'])
        if 'Certainty level' in prediction:
            targets.append(prediction['Certainty level'])
        return ' '.join(targets).lower().strip()


def get_mismatch_output(golds: list, preds: list):
    num = 0
    labels = []
    for pred in preds:
        if pred not in golds:
            num += 1
            labels.append(pred)
    return num, labels


def evaluate():
    path_to_data_file = f"{args.path_to_pred_dir}/raw_outputs/{args.input_file}"
    path_to_eval_file = f"{args.path_to_eval_dir}/{args.input_file}" if not args.allow_mismatch \
        else f"{args.path_to_eval_dir}/[mismatch]_{args.input_file}"

    assert os.path.exists(path_to_data_file), "File Not Exist"

    data = load_dataset(path_to_data_file)
    golds, predictions = [], []
    for utt_id, utt_info in data.items():
        golds.append(utt_info['target'].lower().strip())
        predictions.append(get_pred_value(utt_info['prediction']))

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

    print(f"Output file: {path_to_eval_file}")
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

    task = 'attitude' if 'attitude' in args.input_file else 'certainty'
    evaluate()
