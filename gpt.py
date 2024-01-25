import json
from openai import OpenAI
import argparse
from tqdm import tqdm
from datasets import load_dataset

from data_utils.example_retrieval import example_retrieval
from data_utils.select_data import calculate_sentence_embeddings
from data_utils.instructions import get_prompt_for_task

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def save_data(path_to_data_file: str, data):
    with open(path_to_data_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def format_answer(answer: str):
    try:
        return json.loads(answer)
    except json.decoder.JSONDecodeError:
        return answer


def gpt_predict(prompt, model):
    messages = [{"role": "user", "content": prompt}]
    response = client_api.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )

    answer = format_answer(response.choices[0].message["content"])
    usage = response["usage"]
    return answer, usage


def get_output_in_context(target: str):
    if args.task == 'multitask':
        out_format = f"\nAnswer: {target}."
    elif args.task == 'attitude':
        out_format = f"\nMotivational level: {target}."
    else:
        out_format = f"\nCertainty level: {target}."
    return out_format


def get_output_requirement() -> str:
    if args.task == 'multitask':
        keys = "\"Answer\""
    elif args.task == 'attitude':
        keys = "\"Motivational level\""
    else:
        keys = "\"Certainty level\""
    if args.add_reasoning:
        keys += " and \"Reasons\""
    return f"\nFormat the output as a dict with {keys} as keys."


def add_in_context_examples(sample, train_dataset, train_embs):
    in_context_text = ''
    in_context_targets = []
    selected_examples = []

    text = f"Therapist: {sample['prev_therapist_utt']}\nClient: {sample['client_utterance']}" \
        if args.similarity_unit == 'dialogue' else f"{sample['client_utterance']}"

    test_emb = calculate_sentence_embeddings([text], args)
    selected_indices = example_retrieval(train_embs, test_emb, args)

    for index in selected_indices:
        example_prompt = f"Therapist: {train_dataset[index]['prev_therapist_utt']}" \
                         f"\nClient: \"{train_dataset[index]['client_utterance']}\"" \
            if len(train_dataset[index]['prev_therapist_utt']) > 0 and args.use_therapist_utt \
            else f"Client: \"{train_dataset[index]['client_utterance']}\""
        example_prompt += get_output_in_context(train_dataset[index]['target'])

        selected_examples.append(f"\n\nExample {len(selected_examples) + 1}:\n{example_prompt}")
        in_context_text = ''.join(selected_examples)
        in_context_targets.append(train_dataset[index]['target'])
    return in_context_text, in_context_targets


def experiment():
    # load dataset
    infer_dataset, train_dataset, train_embs = None, None, None

    if args.do_testing:
        infer_dataset = load_dataset(args.dataset_name, args.dataset_config_name)["test"]
    elif args.do_validation:
        infer_dataset = load_dataset(args.dataset_name, args.dataset_config_name)["validation"]
    elif args.input_file is not None:
        with open(f"{args.path_to_data_dir}/{args.input_file}", 'r', encoding='utf-8') as file:
            infer_dataset = json.load(file)
    assert infer_dataset is not None

    to_do_samples = infer_dataset.select(range(args.utt_idx+3)) \
        if args.utt_idx is not None else infer_dataset

    # prepare the prompts
    train_texts = []
    if args.in_context_learning:
        train_dataset = load_dataset(args.dataset_name, args.dataset_config_name)["train"]
        if args.max_training_samples:
            train_dataset = train_dataset.select(range(args.max_training_samples))
        print(f"Dataset length: {len(train_dataset)}")

        for sample in train_dataset:
            text = f"Therapist: {sample['prev_therapist_utt']}\nClient: {sample['client_utterance']}" \
                if args.similarity_unit == 'dialogue' else f"{sample['client_utterance']}"
            train_texts.append(text)

        print(f"Creating embeddings for training samples in ICL...")
        train_embs = calculate_sentence_embeddings(train_texts, args)

    # inference with GPT
    bar = tqdm(range(len(to_do_samples)), desc=f'Predicting with gpt...')
    for sample in to_do_samples:
        bar.update(1)

        if args.do_testing:
            args.output_file = f"{sample['id']}_test_gpt.json"
        elif args.do_validation:
            args.output_file = f"{sample['id']}_validation_gpt.json"
        else:
            args.output_file = f"{sample['id']}_{args.input_file[:-5]}_gpt.json"
        path_to_out_file = f"{args.path_to_output_dir}/{args.output_file}"

        if os.path.exists(path_to_out_file):
            continue
        else:
            therapist = sample['prev_therapist_utt']
            client = sample['client_utterance']
            target = sample['target']
            in_context_targets = []

            if args.in_context_learning:
                in_context_text, in_context_targets = add_in_context_examples(sample, train_dataset, train_embs)
                prompt = get_prompt_for_task(task=args.task, therapist_utt=therapist, client_utt=client,
                                             use_therapist_utt=args.use_therapist_utt, in_context_text=in_context_text,
                                             target_space=target, use_simple_inst=args.use_simplified_instruction)
            else:
                prompt = get_prompt_for_task(task=args.task, therapist_utt=therapist, client_utt=client,
                                             use_therapist_utt=args.use_therapist_utt,
                                             target_space=target, use_simple_inst=args.use_simplified_instruction)
            prompt += get_output_requirement()

            answer, usage = gpt_predict(prompt, args.model_name_or_path)
            prediction = {
                "prev_therapist_utt": therapist,
                "client_utt": client,
                "target": target,
                "prediction": answer,
                "in-context targets": in_context_targets,
                "usage": usage
            }
            save_data(path_to_data_file=path_to_out_file, data={sample['id']: prediction})


def format_output_path():
    args.path_to_output_dir = f"{args.path_to_output_dir}/icl/{args.task}"

    if args.in_context_learning:
        args.path_to_output_dir = f"{args.path_to_output_dir}_in-context"
    if args.use_simplified_instruction:
        args.path_to_output_dir = f"{args.path_to_output_dir}_simplified"

    args.path_to_output_dir = f"{args.path_to_output_dir}/raw_outputs/gpt_{args.dataset_config_name}"

    if args.max_training_samples:
        args.path_to_output_dir = f"{args.path_to_output_dir}-{args.max_training_samples}"

    if args.add_reasoning:
        args.path_to_output_dir = f"{args.path_to_output_dir}_reason"

    if args.in_context_learning:
        args.path_to_output_dir = f"{args.path_to_output_dir}_" \
                                  f"{args.similarity_unit}_{args.num_examples}"

    if not args.use_therapist_utt:
        args.path_to_output_dir = f"{args.path_to_output_dir}_no_therapist_utt"

    if args.example_retrieval_method != 'similarity':
        args.output_file = f"{args.output_file[:-5]}_{args.example_retrieval_method}.json"

    if not os.path.exists(args.path_to_output_dir):
        os.makedirs(args.path_to_output_dir, exist_ok=True)


def merge_files():
    all_preds = {}

    bar = tqdm(range(600), desc=f'Predicting with gpt...')
    for filename in os.listdir(f"{path_to_pred_dir}/{args.dir_to_merge}"):
        path_to_file = f"{path_to_pred_dir}/{args.dir_to_merge}/{filename}"
        bar.update(1)

        if os.path.isfile(path_to_file) and filename.endswith('json'):
            with open(path_to_file, 'r', encoding='utf-8') as file:
                all_preds.update(json.load(file))

    path_to_out_file = f"{path_to_pred_dir}/test_{args.dir_to_merge}.json"
    save_data(path_to_out_file, all_preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="gpt-3.5-turbo")
    parser.add_argument('--openai_key', required=True)
    parser.add_argument('--path_to_data_dir', default='./data')
    parser.add_argument('--path_to_output_dir', default='./outputs_with_certainty')
    parser.add_argument('--input_file', default='test.json')
    parser.add_argument('--dataset_name', type=str, default="./client_mi_dataset.py")
    parser.add_argument('--dataset_config_name', type=str)
    parser.add_argument('--embedding_model', type=str,
                        default='sentence-transformers/paraphrase-mpnet-base-v2')
    parser.add_argument('--task', default='attitude', type=str, choices=['attitude', 'certainty', 'multitask'])
    parser.add_argument('--add_reasoning', action='store_true',
                        help='Whether we want the model to explain their reasoning or not.')
    parser.add_argument('--use_therapist_utt', action='store_true',
                        help='Whether to include the therapist utt as context in the dialogue or not.')
    parser.add_argument('--in_context_learning', action='store_true',
                        help='Whether to use do in context learning for inference.')
    parser.add_argument('--use_simplified_instruction', action='store_true',
                        help='Whether to use output space without explanations.')
    parser.add_argument('--do_testing', action='store_true',
                        help='Whether to use the test set.')
    parser.add_argument('--do_validation', action='store_true',
                        help='Whether to use the validation set.')
    parser.add_argument('--example_retrieval_method', type=str, default='similarity',
                        choices=['similarity', 'random'])
    parser.add_argument('--similarity_unit', type=str, default='dialogue',
                        choices=['utterance', 'dialogue'])
    parser.add_argument('--num_examples', type=int, default=1,  # to investigate
                        help='Number of examples used in in-context learning setting.')
    parser.add_argument('--max_training_samples', type=int,  # 50, 100, 200, 300
                        help='Maximum number of examples to choose in in-context learning setting.')
    parser.add_argument('--utt_idx', type=int)
    parser.add_argument('--job', default='experiment', choices=['experiment', 'merge'], type=str)
    parser.add_argument('--dir_to_merge', type=str, default=None)
    args = parser.parse_args()

    if args.job == 'experiment':
        client_api = OpenAI()
        format_output_path()
        OpenAI.api_key = args.openai_key
        experiment()
    else:
        if args.dir_to_merge:
            path_to_pred_dir = f"{args.path_to_output_dir}/icl/multitask/raw_outputs"
            merge_files()
        else:
            print(f"Value dir_to_merge can't be empty.")
