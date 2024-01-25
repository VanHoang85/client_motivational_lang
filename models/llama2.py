# https://huggingface.co/blog/llama2

from huggingface_hub import login
from transformers import LlamaTokenizer
import transformers
import argparse
# import torch


def get_llama(model_name: str):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    return model_name, tokenizer


def llama_predict(prompt: str, model, tokenizer, args):

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        device_map="auto",
        # torch_dtype=torch.float16,
        model_kwargs={"load_in_8bit": True}
    )

    system_message = "You are a clinical psychologist using motivational interviewing technique."
    prompt = f"[INST] <<SYS>>{system_message}<</SYS>>\n\n {prompt} [/INST]"

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
    # for seq in sequences:
    #     print(f"Result: {seq['generated_text']}")
    return sequences[0]['generated_text']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2data', default='../AnnoMI-full-utt.json')
    parser.add_argument('--model_name_or_path,', default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument('--max_length', type=int, default=500)
    args = parser.parse_args()

    login(token='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')

    #  run time: "meta-llama/Llama-2-13b-chat-hf" --> 100 samples 1h'
    #  run time: "meta-llama/Llama-2-13b-chat-hf" --> 300 samples 3h18m'
