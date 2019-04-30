#!/usr/bin/env bash
python sct.py \
-m train \
-id base_0429_30k \
-model base \
-cuda 2,3 \
-injson data/sct_caps2idx_30k.json \
-infeatdir data/features_30k_14 \
-incaph5 data/sct_caps_30k.h5 \
-outmemdir data/memory \
-inmemh5 data/memory/memory_base_0429_30k.h5 \
-tm 1 \
-tdir log/base_0429_30k \
-chkpt log/base_0429_30k \
-laneval 1 \
-batch 128 \
-batch3 32 \
-savechkpteve 450 \
-savechkpteve3 500 \
-epo 80 \
-epo3 20 \
-lr 5e-5 \
-fusecoef 0.50 \
-best1 data/best1 \
-best3 data/best3 \
-start log/base_0429_30k