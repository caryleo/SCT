#!/usr/bin/env bash
python sct.py \
-m train \
-id base_0523 \
-model base \
-cuda 0,1,2,3 \
-injson data/sct_coco_100.json \
-infeatdir data/features_14 \
-incaph5 data/sct_coco_100.h5 \
-outmemdir data/memory \
-inmemh5 data/memory/memory_base_0523.h5 \
-tm 1 \
-tdir log/base_0523 \
-chkpt log/base_0523 \
-laneval 1 \
-batch 256 \
-batch3 64 \
-savechkpteve 300 \
-savechkpteve3 1000 \
-epo 25 \
-epo3 20 \
-best1 data/best1 \
-best3 data/best3 \
-lr 4e-4 \
-fusecoef 0.25 \
-rebest 0
# -start log/base_0513