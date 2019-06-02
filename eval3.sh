#!/usr/bin/env bash
#python sct.py \
#-m eval \
#-images 5000 \
#-modpth data/best3/model_base_0521-best3.pth \
#-em 3 \
#-cuda 1 \
#-infopth data/best3/info_base_0521-best.pkl \
#-inmemh5 data/memory/memory_base_0521.h5 \
#-injson data/sct_coco_100.json \
#-s few-test \
#-met 1 \
#-laneval 1

python sct.py \
-m eval \
-images 5000 \
-modpth log/base_0521/model-best.pth \
-em 3 \
-cuda 1 \
-infopth log/base_0521/info_base_0521-best.pkl \
-inmemh5 data/memory/memory_base_0521.h5 \
-s test \
-met 4 \
-laneval 1
