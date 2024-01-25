# templates: https://github.com/google-research/FLAN/blob/main/flan/v2/flan_templates_branched.py
# https://huggingface.co/docs/transformers/model_doc/flan-t5

import torch
import argparse
from typing import Union, List
from transformers import T5Tokenizer, T5ForConditionalGeneration


def t5_predict(prompts: Union[str, List[str]], model, tokenizer, args):
    inputs = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True).to(args.device)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=False,  # disable sampling to test if batching affects output
        max_new_tokens=args.max_length,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)  # return List[str]


def get_t5(model_name: str, args):
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        # device_map="auto",  # auto will distribute memory on multiple device
        torch_dtype=torch.float16,
        # torch_dtype="auto",
        load_in_8bit=True,
        # load_in_4bit=True,
    )
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2data', default='../AnnoMI-full-dialog.json')
    parser.add_argument('--model_name_or_path,', default="google/flan-t5-xxl")  # 11B parameters, ~20GB
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # .to(args.device)
    # max tokens = 512
