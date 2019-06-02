#!/usr/bin/env bash
python sct.py \
-m eval \
-cuda 3 \
-modpth data/best3/model_base_0511-best3.pth \
-infopth data/best3/info_base_0511-best.pkl \
-inmemh5 data/memory/memory_base_0511.h5 \
-images 50 \
-img test_rare_images \
-dumpimg 1 \
-dumpjson 1