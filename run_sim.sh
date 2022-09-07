#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 \
python3 simulator.py \
--model_name_or_path ./bigdataset_base/checkpoint-20000 \
--max_len 50 \
--split test \
--top_p 0.8 \
--temp 1.0 \
--num_chats 980 \
--disable_output_dialog \
--output ./final/output_p{0.8}_t{1.0}.jsonl
