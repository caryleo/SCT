#!/usr/bin/env bash
python sct.py \
-m eval \
-images 5000 \
-modpth data/best3/model_finetune_0515-best3.pth \
-em 3 \
-cuda 3 \
-infopth data/best3/info_finetune_0515-best.pkl \
-inmemh5 data/memory/memory_finetune_0515.h5 \
-s test \
-met 4 \
-laneval 1

#python sct.py \
#-m eval \
#-images 5000 \
#-modpth log/base_0509/model-best.pth \
#-em 3 \
#-cuda 6 \
#-infopth log/base_0509/info_base_0509-best.pkl \
#-inmemh5 data/memory/memory_base_0509.h5 \
#-s few-test \
#-met 4 \
#-laneval 1
