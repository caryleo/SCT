#!/usr/bin/env bash
python sct.py \
-m eval \
-images 5000 \
-modpth data/best1/model_base_0601_8k-best1.pth \
-em 1 \
-cuda 7 \
-infopth data/best1/info_base_0601_8k-best.pkl \
-inmemh5 data/memory/memory_base_0601_8k.h5 \
-injson data/sct_flickr8k_all.json \
-s test \
-met 1 \
-laneval 1 \
-datas flickr8k
