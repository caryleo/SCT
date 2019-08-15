#!/usr/bin/env bash
python sct.py \
-m eval \
-images 5000 \
-modpth data/best3/model_base_0801_30-best3.pth \
-em 3 \
-cuda 1 \
-infopth data/best3/info_base_0801_30-best.pkl \
-inmemh5 data/memory/memory_base_0801_30.h5 \
-injson data/sct_flickr30k_all.json \
-s test \
-met 1 \
-laneval 1 \
-datas flickr30k
