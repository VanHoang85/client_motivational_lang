import os
import gc
import json
import logging
import argparse
import torch
import openai
from tqdm import tqdm
from typing import Union, List
from datasets import load_dataset, Dataset
from huggingface_hub import login

from gpt import gpt_predict
from models.llama2 import get_llama, llama_predict
from models.flanT5 import get_t5, t5_predict
from models.falcon import get_falcon, falcon_predict

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel, PeftConfig

from data_utils.instructions import get_prompt_for_task
from data_utils.select_data import calculate_sentence_embeddings
from data_utils.example_retrieval import example_retrieval


def save_data(path_to_data_file: str, data):
    with open(path_to_data_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def get_models():
    model, tokenizer = None, None

    # merge the LORA weights for fast inference
    if 'merged' not in args.model_name_or_path and 'lora' in args.model_name_or_path:
        checkpoint = get_last_checkpoint(args.model_name_or_path)
        peft_config = PeftConfig.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path,
                                                      low_cpu_mem_usage=True,
                                                      torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

        args.model_name_or_path = f"{args.model_name_or_path}_merged"
        model.save_pretrained(args.model_name_or_path)
        tokenizer.save_pretrained(args.model_name_or_path)

    # load models
    if 'peft' in args.model_name_or_path and 'merged' not in args.model_name_or_path:
        checkpoint = get_last_checkpoint(args.model_name_or_path)
        peft_config = PeftConfig.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path,
                                                      low_cpu_mem_usage=True,
                                                      torch_dtype=torch.float16,
                                                      load_in_8bit=True)
        model = PeftModel.from_pretrained(model, checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    else:
        if 't5' in args.model_name_or_path:
            model, tokenizer = get_t5(args.model_name_or_path, args)
        elif 'falcon' in args.model_name_or_path:
            model, tokenizer = get_falcon(args.model_name_or_path)
        elif 'llama' in args.model_name_or_path:
            model, tokenizer = get_llama(args.model_name_or_path)
        elif 'gpt' in args.model_name_or_path:
            model = args.model_name_or_path

    model.eval()
    return model, tokenizer


def make_prediction(prompt: Union[str, List[str]], model, tokenizer):
    answers = None
    if 'gpt' in args.model_name_or_path:
        answers, usage = gpt_predict(prompt, model)
        total_usages.append(usage)
        total_tokens.append(int(usage['total_tokens']))

    elif 't5' in args.model_name_or_path:
        answers = t5_predict(prompt, model, tokenizer, args)

    elif 'falcon' in args.model_name_or_path:
        answers = falcon_predict(prompt, model, tokenizer, args)
    elif 'llama' in args.model_name_or_path:
        answers = llama_predict(prompt, model, tokenizer, args)
    return answers


def add_in_context_examples(sample, train_dataset, train_embs):
    in_context_text = ''
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
        example_prompt += f"\nAnswer: {train_dataset[index]['target']}"

        selected_examples.append(f"\n\nExample {len(selected_examples) + 1}:\n{example_prompt}")
        in_context_text = ''.join(selected_examples)
    return in_context_text


def experiment():
    infer_dataset, train_dataset, train_embs = None, None, None

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
        train_embs = calculate_sentence_embeddings(train_texts, args)

    if args.input_file is not None:
        with open(f"{args.path_to_data_dir}/{args.input_file}", 'r', encoding='utf-8') as file:
            raw_dataset = json.load(file)

        def data_generator():
            for utt_id, utt_info in raw_dataset.items():
                if utt_info['utt_info']['interlocutor'] == 'client':
                    yield {
                        "id": utt_id,
                        "client_utterance": utt_info['utt_info']['text'],
                        "prev_therapist_utt": utt_info['utt_info']['prev_utt'],
                        "target": "",
                    }
        infer_dataset = Dataset.from_generator(data_generator)

    elif args.do_testing:
        infer_dataset = load_dataset(args.dataset_name, args.dataset_config_name)["test"]
    elif args.do_validation:
        infer_dataset = load_dataset(args.dataset_name, args.dataset_config_name)["validation"]
    assert infer_dataset is not None

    model, tokenizer = get_models()

    bar = tqdm(range(len(infer_dataset)), desc=f'Processing prompts...')
    processed_data = []
    for sample in infer_dataset:
        therapist = sample['prev_therapist_utt']
        client = sample['client_utterance']
        target = sample['target']

        # if do in-context learning, get examples from train dataset here
        if args.in_context_learning:
            in_context_text = add_in_context_examples(sample, train_dataset, train_embs)
            prompt = get_prompt_for_task(task=args.task, therapist_utt=therapist, client_utt=client,
                                         use_therapist_utt=args.use_therapist_utt, in_context_text=in_context_text,
                                         target_space=target, use_simple_inst=args.use_simplified_instruction)
        else:
            prompt = get_prompt_for_task(task=args.task, therapist_utt=therapist, client_utt=client,
                                         use_therapist_utt=args.use_therapist_utt,
                                         target_space=target, use_simple_inst=args.use_simplified_instruction)
        prompt += f"\nAnswer:"
        # print(prompt)
        processed_data.append(prompt)
        bar.update(1)

    predictions = []
    bar = tqdm(range(len(processed_data)), desc=f'Inference with {args.model_name_or_path}...')
    for idx in range(0, len(processed_data), args.inf_batch_size):
        prompts = processed_data[idx: idx + args.inf_batch_size]
        answers = make_prediction(prompts, model, tokenizer)
        predictions.extend(answers)
        bar.update(args.inf_batch_size)

    outputs = {}
    for sample, pred in zip(infer_dataset, predictions):
        item = {
            "prev_therapist_utt": sample['prev_therapist_utt'],
            "client_utt": sample['client_utterance'],
            "target": sample['target'],
            "prediction": pred,
        }
        outputs[sample['id']] = item

    print(f"{args.path_to_output_dir}/{args.output_file}")
    save_data(path_to_data_file=f"{args.path_to_output_dir}/{args.output_file}", data=outputs)


def format_output_path():
    args.path_to_output_dir = f"{args.path_to_output_dir}/icl/{args.task}"
    if args.in_context_learning:
        args.path_to_output_dir = f"{args.path_to_output_dir}_in-context"

    if args.use_simplified_instruction:
        args.path_to_output_dir = f"{args.path_to_output_dir}_simplified"

    args.path_to_output_dir = f"{args.path_to_output_dir}/raw_outputs"

    if not os.path.exists(args.path_to_output_dir):
        os.makedirs(args.path_to_output_dir, exist_ok=True)

    if 'peft' not in args.model_name_or_path:
        args.output_file = f"{args.model_name_or_path.split('/')[-1]}_{args.dataset_config_name}"
    else:
        args.output_file = f"{args.model_name_or_path.split('/')[-1]}"

    if args.input_file is not None:
        args.output_file = f"{args.input_file[:-5]}_{args.output_file}.json"
    elif args.do_testing:
        args.output_file = f"test_{args.output_file}.json"
    elif args.do_validation:
        args.output_file = f"validation_{args.output_file}.json"

    if args.max_training_samples:
        args.output_file = f"{args.output_file[:-5]}-{args.max_training_samples}.json"

    if args.add_reasoning:
        args.output_file = f"{args.output_file[:-5]}_reasoning.json"

    if args.in_context_learning:
        args.output_file = f"{args.output_file[:-5]}_" \
                           f"{args.similarity_unit}_{args.num_examples}.json"

    if args.example_retrieval_method != 'similarity':
        args.output_file = f"{args.output_file[:-5]}_{args.example_retrieval_method}.json"

    if not args.use_therapist_utt:
        args.output_file = f"{args.output_file[:-5]}_no_therapist_utt.json"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data_dir', default='./data')
    parser.add_argument('--path_to_output_dir', default='./outputs_with_certainty')
    parser.add_argument('--input_file', type=str)  # eg, test.json
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--dataset_config_name', type=str)
    parser.add_argument('--hf_key', required=True)
    parser.add_argument('--embedding_model', type=str,
                        default='sentence-transformers/paraphrase-mpnet-base-v2')
    parser.add_argument('--task', default='attitude', choices=['attitude', 'certainty', 'multitask'])
    parser.add_argument('--add_reasoning', action='store_true',
                        help='Whether we want the model to explain their reasoning or not.')
    parser.add_argument('--use_therapist_utt', action='store_true',
                        help='Whether to include the therapist utt as context in the dialogue or not.')
    parser.add_argument('--in_context_learning', action='store_true',
                        help='Whether to do in context learning for inference.')
    parser.add_argument('--use_simplified_instruction', action='store_true',
                        help='Whether to use output space without explanations.')
    parser.add_argument('--do_testing', action='store_true',
                        help='Whether to use the test set.')
    parser.add_argument('--do_validation', action='store_true',
                        help='Whether to use the validation set.')
    parser.add_argument('--inf_batch_size', type=int, default=16)
    parser.add_argument('--example_retrieval_method', type=str, default='similarity',
                        choices=['similarity', 'random'])
    parser.add_argument('--similarity_unit', type=str, default='dialogue',
                        choices=['utterance', 'dialogue'])
    parser.add_argument('--num_examples', type=int, default=1,
                        help='Number of examples used in in-context learning setting.')
    parser.add_argument('--max_training_samples', type=int,  # 50, 100, 200, 300
                        help='Maximum number of examples to choose in in-context learning setting.')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--return_full_text', action='store_true')
    parser.add_argument('--repetition_penalty', type=float, default=1.1)
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = logging.getLogger(__name__)
    logger.info(args.device)

    if 'llama' in args.model_name_or_path:
        login(token=args.hf_key)
        args.model_type = 'llama'
    elif 'gpt' in args.model_name_or_path:
        openai.api_key = args.openai_key
        args.model_type = 'gpt'
    elif 't5' in args.model_name_or_path:
        args.model_type = 'flan-t5'
    elif 'falcon' in args.model_name_or_path:
        args.model_type = 'falcon'

    # to track gpt_reason_commit usage
    total_usages, total_tokens = [], []

    format_output_path()
    experiment()

    torch.cuda.empty_cache()
    gc.collect()

    # if use gpt_reason_commit, print total tokens
    if len(total_tokens) > 0:
        print(f"Total number of tokens for the experiment: {sum(total_tokens)}")
