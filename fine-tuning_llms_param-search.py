#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import datasets
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score

import wandb
from data_utils.instructions import get_prompt_for_task

from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    AdaLoraConfig,
    LoraConfig,
    IA3Config,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig,
)

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    default_data_collator,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.15.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    input_file: str = field(
        default=None, metadata={"help": "Path to file to do prediction with the classifier."}
    )
    max_input_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_output_seq_length: int = field(
        default=3,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    pred_dir: str = field(
        default=None,
        metadata={"help": "Path to outputs dir."}
    )
    output_file: str = field(default=None)
    use_instruction: bool = field(default=False)
    use_simplified_instruction: bool = field(default=False)
    use_therapist_utt: bool = field(default=False)
    max_training_samples: int = field(default=None)  # 50, 100, 200, 300
    task: str = field(default='attitude', metadata={"help": "choose between attitude and certainty, or both"})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    num_early_stopping: int = field(
        default=3,
        metadata={
            "help": "The maximum total number of times validation metrics failed to improve."
        },
    )
    peft_method: str = field(
        default=None,
        metadata={"help": "Which peft method to use for fine tuning. If None, fine tuning without PEFT."}
    )
    num_virtual_tokens: int = field(
        default=20,
        metadata={"help": "The number of virtual tokens to use, or in other words, the prompt."}
    )
    lora_r: int = field(
        default=16,  # 16
        metadata={"help": "the rank of the update matrices, expressed in int. "
                          "Lower rank results in smaller update matrices with fewer trainable parameters."}
    )
    lora_init_r: int = field(default=16)
    lora_target_r: int = field(default=12)
    lora_alpha: int = field(
        default=32,  # 32
        metadata={"help": "LoRA scaling factor."}
    )
    lora_beta1: float = field(default=0.85)
    lora_beta2: float = field(default=0.85)
    lora_tinit: int = field(default=200)
    lora_tfinal: int = field(default=1000)
    lora_delta: int = field(default=10)
    lora_dropout: float = field(
        default=0.1,  # 16
        metadata={"help": ""}
    )
    lora_train_all_layers: bool = field(
        default=False,
        metadata={"help": "Whether to train all linear layers or just the default attention layers."}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load and fine-tune the models in 8-bit."}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to load and fine-tune the models in 4-bit."}
    )
    num_trial_search: int = field(
        default=10,
        metadata={"help": "The number of trial to do hyper-parameter search."}
    )
    max_lr: float = field(
        default=1e-3,
    )
    min_lr: float = field(
        default=5e-4,
    )
    project_name: str = field(
        default='search'
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)

    # create output and cache dirs if not exist
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
    if not os.path.exists(model_args.cache_dir):
        os.makedirs(model_args.cache_dir, exist_ok=True)

    if training_args.do_predict:
        data_args.output_file = f"test_{model_args.model_name_or_path}_{data_args.dataset_config_name}.json"
    elif training_args.do_eval:
        data_args.output_file = f"validation_{model_args.model_name_or_path}_{data_args.dataset_config_name}.json"

    if model_args.peft_method is not None:
        data_args.output_file = f"{data_args.output_file[:-5]}_{model_args.peft_method}.json"

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)

    target_space = raw_datasets['test'][0]["target"].split(' ')
    if data_args.task == 'multitask':
        assert len(target_space) == 3, f"target space is incorret: {target_space}"
    else:
        assert len(target_space) == 1, f"target space is incorret: {target_space}"

    # ---------------------------------------------------------------------------------------------------
    # ------------------------------- Load pretrained MODEL AND TOKENIZER -------------------------------
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    def get_peft_utils(model):
        peft_config = None
        task_type = TaskType.SEQ_2_SEQ_LM
        training_args.output_dir = f"{training_args.output_dir}/" \
                                   f"{model_args.model_name_or_path}_{data_args.dataset_config_name}"
        if data_args.max_training_samples:
            training_args.output_dir = f"{training_args.output_dir}-{data_args.max_training_samples}"

        if model_args.peft_method == 'p-tuning':
            peft_config = PromptEncoderConfig(task_type=task_type,
                                              num_virtual_tokens=model_args.num_virtual_tokens,
                                              encoder_hidden_size=128)
            training_args.output_dir = f"{training_args.output_dir}-peft-p-tuning"
        elif model_args.peft_method == 'prompt-tuning':
            peft_config = PromptTuningConfig(task_type=task_type,
                                             num_virtual_tokens=model_args.num_virtual_tokens)
            training_args.output_dir = f"{training_args.output_dir}-peft-prompt-tuning"
        elif model_args.peft_method == 'prefix-tuning':
            peft_config = PrefixTuningConfig(task_type=task_type,
                                             num_virtual_tokens=model_args.num_virtual_tokens)
            training_args.output_dir = f"{training_args.output_dir}-peft-prefix-tuning"
        elif model_args.peft_method == 'lora':
            if model_args.lora_train_all_layers:
                target_modules = [name for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
                peft_config = LoraConfig(task_type=task_type,
                                         inference_mode=False,
                                         r=model_args.lora_r,
                                         lora_alpha=model_args.lora_alpha,
                                         lora_dropout=model_args.lora_dropout,
                                         target_modules=target_modules)
            else:
                peft_config = LoraConfig(task_type=task_type,
                                         inference_mode=False,
                                         r=model_args.lora_r,
                                         lora_alpha=model_args.lora_alpha,
                                         lora_dropout=model_args.lora_dropout)
            training_args.output_dir = f"{training_args.output_dir}-peft-lora"

        elif model_args.peft_method == 'adalora':
            if model_args.lora_train_all_layers:
                target_modules = [name for name, layer in model.named_modules() if isinstance(layer, torch.nn.Linear)]
                peft_config = AdaLoraConfig(task_type=task_type,
                                            inference_mode=False,
                                            init_r=model_args.lora_init_r,
                                            target_r=model_args.lora_target_r,
                                            beta1=model_args.lora_beta1,
                                            beta2=model_args.lora_beta2,
                                            tinit=model_args.lora_tinit,
                                            deltaT=model_args.lora_delta,
                                            lora_alpha=model_args.lora_alpha,
                                            lora_dropout=model_args.lora_dropout,
                                            target_modules=target_modules)
            else:
                peft_config = AdaLoraConfig(task_type=task_type,
                                            inference_mode=False,
                                            init_r=model_args.lora_init_r,
                                            target_r=model_args.lora_target_r,
                                            beta1=model_args.lora_beta1,
                                            beta2=model_args.lora_beta2,
                                            tinit=model_args.lora_tinit,
                                            deltaT=model_args.lora_delta,
                                            lora_alpha=model_args.lora_alpha,
                                            lora_dropout=model_args.lora_dropout)

            training_args.output_dir = f"{training_args.output_dir}-peft-adalora"
            if model_args.lora_train_all_layers:
                training_args.output_dir = f"{training_args.output_dir}-peft-adalora-all-layers"

        elif model_args.peft_method == 'ia3':
            peft_config = IA3Config(task_type=task_type,
                                    inference_mode=False)
            training_args.output_dir = f"{training_args.output_dir}-peft-ia3"
        return peft_config

    if model_args.load_in_8bit:
        def model_init(trial):
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                load_in_8bit=True,
            )

            """### Post-processing on the model
                        Finally, we need to apply some post-processing on the 8-bit model to enable training, 
                        let's freeze all our layers, and cast the layer-norm in `float32` for stability. 
                        We also cast the output of the last layer in `float32` for the same reasons.
                        """
            for param in model.parameters():
                param.requires_grad = False  # freeze the model - train adapters later
                if param.ndim == 1:
                    # cast the small parameters (e.g. layernorm) to fp32 for stability
                    param.data = param.data.to(torch.float32)

            class CastOutputToFloat(torch.nn.Sequential):
                def forward(self, x):
                    return super().forward(x).to(torch.float32)

            model.lm_head = CastOutputToFloat(model.lm_head)

            peft_config = get_peft_utils(model)
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)
            return model

    elif model_args.load_in_4bit:
        def model_init(trial):
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            )
            peft_config = get_peft_utils(model)
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)
            return model

    # ---------------------------------------------------------------------------------------------------
    # ----------------------------------------- DATA PROCESSING -----------------------------------------
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    max_input_seq_length = min(data_args.max_input_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        if data_args.use_instruction:
            dialogues = []
            for therapist_utt, client_utt, target in \
                    zip(examples['prev_therapist_utt'], examples['client_utterance'], examples['target']):
                prompt = get_prompt_for_task(data_args.task, therapist_utt, client_utt,
                                             use_therapist_utt=data_args.use_therapist_utt, target_space=target)
                dialogues.append(f"{prompt}\nAnswer:")
        else:
            dialogues = [f"Therapist: {therapist_utt}\nClient: \"{client_utt}\"" for therapist_utt, client_utt
                         in zip(examples['prev_therapist_utt'], examples['client_utterance'])]
        result = tokenizer(dialogues, padding=padding, max_length=max_input_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if "target" in examples:
            labels = tokenizer(text_target=examples["target"], padding=padding,
                               max_length=data_args.max_output_seq_length, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to
            # ignore padding in the loss.
            labels["input_ids"] = [[(label if label != tokenizer.pad_token_id else -100)
                                    for label in label] for label in labels["input_ids"]]
            result["labels"] = labels["input_ids"]
        return result

    train_dataset, eval_dataset = raw_datasets["train"], None
    if data_args.max_training_samples:
        train_dataset = train_dataset.select(range(data_args.max_training_samples))
    print(f"Dataset length: {len(train_dataset)}")

    if training_args.do_train:
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if model_args.load_in_8bit or model_args.load_in_4bit:
        data_collator = DataCollatorWithPadding(
            tokenizer,
            padding=padding,
            pad_to_multiple_of=8,
            return_tensors='pt',
        )
    else:
        data_collator = default_data_collator

    # ---------------------------------------------------------------------------------------------------
    # ------------------------------------------- DO TRAINING -------------------------------------------
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def format_outputs(eval_preds):
        preds = eval_preds.predictions[0] if isinstance(eval_preds.predictions, tuple) else eval_preds.predictions
        labels = eval_preds.label_ids

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        preds = [pred.strip().lower() for pred in decoded_preds]
        labels = [label.strip().lower() for label in decoded_labels]
        return preds, labels

    def compute_metrics(eval_preds: EvalPrediction):
        preds, labels = format_outputs(eval_preds)
        return {"accuracy": accuracy_score(y_true=labels, y_pred=preds),
                "f1": f1_score(y_true=labels, y_pred=preds, average="macro")}

    def format_multitask_outputs(eval_preds):
        preds = eval_preds.predictions[0] if isinstance(eval_preds.predictions, tuple) else eval_preds.predictions
        labels = eval_preds.label_ids

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        out_preds, out_labels = {}, {}
        for pred, label in zip(decoded_preds, decoded_labels):
            label = label.strip().lower()
            pred = pred.strip().lower()

            if len(label) == 1 and label.strip().lower() in ['change', 'neutral', 'sustain']:
                if 'att' not in out_labels:
                    out_labels['att'] = []
                    out_preds['att'] = []
                out_labels['att'].append(label)
                out_preds['att'].append(pred)

            elif len(label) == 1 and label.strip().lower() in ['high', 'medium', 'low']:
                if 'cer' not in out_labels:
                    out_labels['cer'] = []
                    out_preds['cer'] = []
                out_labels['cer'].append(label)
                out_preds['cer'].append(pred)

            else:
                if 'multi' not in out_labels:
                    out_labels['multi'] = []
                    out_preds['multi'] = []
                out_labels['multi'].append(label)
                out_preds['multi'].append(pred)
        return out_preds, out_labels

    def compute_multitask_metrics(eval_preds: EvalPrediction):
        preds, labels = format_multitask_outputs(eval_preds)

        acc_att = accuracy_score(y_true=labels['att'], y_pred=preds['att'])
        acc_cer = accuracy_score(y_true=labels['cer'], y_pred=preds['cer'])
        acc_mul = accuracy_score(y_true=labels['multi'], y_pred=preds['multi'])
        f1_att = f1_score(y_true=labels['att'], y_pred=preds['att'], average="macro")
        f1_cer = f1_score(y_true=labels['cer'], y_pred=preds['cer'], average="macro")
        f1_mul = f1_score(y_true=labels['mul'], y_pred=preds['mul'], average="macro")
        return {"accuracy": (acc_att + acc_cer + acc_mul) / 3,
                "f1": (f1_att + f1_cer + f1_mul) / 3,
                "acc_att": acc_att,
                "acc_cer": acc_cer,
                "acc_multi": acc_mul,
                "f1_att": f1_att,
                "f1_cer": f1_cer,
                "f1_multi": f1_mul}

    sweep_config = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {
            'goal': 'maximize',
            'name': 'val_f1'
        },
        'parameters': {
            'batch_size': {'values': [4, 6, 8]},
            'epochs': {'values': [30]},
            'weight_decay': {'values': [0.0, 1e-6]},
            'learning_rate': {'max': model_args.max_lr, 'min': model_args.min_lr}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=model_args.project_name)

    def train(config=None):
        with wandb.init(config=config):
            # set sweep configuration
            config = wandb.config

            # set training arguments
            training_args = Seq2SeqTrainingArguments(
                output_dir='./search-t5',
                report_to='wandb',  # Turn on Weights & Biases logging
                num_train_epochs=config.epochs,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=16,
                optim='adafactor',
                save_strategy='epoch',
                evaluation_strategy='epoch',
                logging_strategy='epoch',
                # logging_steps=500,
                # save_steps=500,
                load_best_model_at_end=True,
                overwrite_output_dir=True,
                ddp_find_unused_parameters=False,
                predict_with_generate=True,
            )

            trainer = Seq2SeqTrainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics if data_args.task != 'multitask' else compute_multitask_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=model_args.num_early_stopping)]
            )

            # start training loop
            trainer.train()

    wandb.agent(sweep_id, train, count=model_args.num_trial_search)


if __name__ == "__main__":
    main()
