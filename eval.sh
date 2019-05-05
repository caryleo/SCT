#!/usr/bin/env bash
python sct.py \
-m eval \
-images 5000 \
-modpth data/best3/model_base_0501-best3.pth \
-em 3 \
-cuda 2,3 \
-infopth data/best3/info_base_0501-best.pkl \
-inmemh5 data/memory/memory_base_0501.h5 \
-injson data/sct_coco_10.json \
-s few-test \
-met 1 \
-laneval 1
