#!/bin/sh

python3 llm_inference.py \
 --model_name_or_path "path_to_model_folder" \
 --input_file "AnnoMI-full-utt.json" \
 --task "certainty" \
 --use_simplified_instruction \
 --inf_batch_size 16 \
