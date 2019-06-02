#!/usr/bin/env bash
python sct.py \
-m eval \
-images 5000 \
-modpth data/best1/model_base_0521-best1.pth \
-em 1 \
-cuda 5 \
-infopth data/best1/info_base_0521-best.pkl \
-inmemh5 data/memory/memory_base_0521.h5 \
-s test \
-met 1 \
-laneval 1
