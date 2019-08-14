#!/usr/bin/env bash
python sct.py \
-m eval \
-images 5000 \
-modpth data/best3/model_base_0525-best3.pth \
-em 3 \
-cuda 2 \
-infopth data/best3/info_base_0525-best.pkl \
-inmemh5 data/memory/memory_base_0525.h5 \
-s test \
-met 1 \
-laneval 1 \
-datas flickr8k
