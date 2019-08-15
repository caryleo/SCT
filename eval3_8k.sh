#!/usr/bin/env bash
#python sct.py \
#-m eval \
#-images 5000 \
#-modpth data/best3/model_base_0811_8k-best3.pth \
#-em 3 \
#-cuda 0 \
#-infopth data/best3/info_base_0811_8k-best.pkl \
#-inmemh5 data/memory/memory_base_0811_8k.h5 \
#-injson data/sct_flickr8k_all.json \
#-s test \
#-met 1 \
#-laneval 1 \
#-datas flickr8k

python sct.py \
-m eval \
-images 5000 \
-modpth log/base_0811_8k/model-best.pth \
-em 3 \
-cuda 1 \
-infopth log/base_0811_8k/info_base_0811_8k-best.pkl \
-inmemh5 data/memory/memory_base_0811_8k.h5 \
-s test \
-met 1 \
-laneval 1 \
-datas flickr8k