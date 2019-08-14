#!/usr/bin/env bash
python sct.py \
-m train \
-id base_0801_30 \
-model base \
-cuda 1,2 \
-injson data/sct_flickr30k_50.json \
-infeatdir data/features_30k_14 \
-incaph5 data/sct_flickr30k_50.h5 \
-outmemdir data/memory \
-inmemh5 data/memory/memory_base_0801_30.h5 \
-tm 1 \
-tdir log/base_0801_30 \
-chkpt log/base_0801_30 \
-laneval 1 \
-batch 128 \
-batch3 32 \
-savechkpteve 120 \
-savechkpteve3 500 \
-epo 70 \
-epo3 30 \
-best1 data/best1 \
-best3 data/best3 \
-lr 5e-4 \
-fusecoef 0.15 \
-rebest 1 \
-datas flickr30k