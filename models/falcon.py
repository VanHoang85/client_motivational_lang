# https://huggingface.co/blog/falcon
# https://huggingface.co/docs/transformers/main/model_doc/falcon#falcon
# https://lightning.ai/blog/falcon-a-guide-to-finetune-and-inference/

import argparse
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


def falcon_predict(prompt: str, model, tokenizer, args):

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    sequences = pipeline(
        prompt,
        do_sample=args.do_sample,
        top_k=args.top_k,
        num_return_sequences=args.num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        max_length=args.max_length,
        return_full_text=args.return_full_text,
        repetition_penalty=args.repetition_penalty  # without this output begins repeating
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

    return sequences[0]['generated_text']


def get_falcon(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_8bit=True,
        device_map="auto",
    )
    return model, tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2data', default='../AnnoMI-full-utt.json')
    parser.add_argument('--model_name_or_path,', default="tiiuae/falcon-7b-instruct")  # falcon 7B instruct ~15GB
    parser.add_argument('--max_length', default=500)
    args = parser.parse_args()

    # run time: 7b-instruct --> 5 samples 2m
    # run time: 7b-instruct --> 100 samples 32h
