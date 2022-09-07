#!/bin/bash

mkdir model
cd model
wget https://www.dropbox.com/s/0ipamifwnbrf4ub/config.json?dl=1 -O config.json
wget https://www.dropbox.com/s/hnr6nitn5dbxmlu/pytorch_model.bin?dl=1 -O pytorch_model.bin
wget https://www.dropbox.com/s/2vizubkgwt16dcf/special_tokens_map.json?dl=1 -O special_tokens_map.json
wget https://www.dropbox.com/s/nihmwok7omb8g17/tokenizer_config.json?dl=1 -O tokenizer_config.json
wget https://www.dropbox.com/s/0teaff8wxi9amy2/tokenizer.json?dl=1 -O tokenizer.json
cd ..

#CUDA_VISIBLE_DEVICES=$1 \


python3 simulator.py \
--model_name_or_path ./model \
--max_len 50 \
--split test \
--num_beams 30 \
--num_chats 980 \
--disable_output_dialog \
--output ./output.jsonl
