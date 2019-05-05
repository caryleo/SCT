#!/usr/bin/env bash
python sct.py \
-m train \
-id base_0503 \
-model base \
-cuda 4,5,6,7 \
-injson data/sct_caps2idx.json \
-infeatdir data/features_14 \
-incaph5 data/sct_caps.h5 \
-outmemdir data/memory \
-inmemh5 data/memory/memory_base_0503.h5 \
-tm 2 \
-tdir log/base_0503 \
-chkpt log/base_0504 \
-laneval 1 \
-batch 256 \
-batch3 64 \
-savechkpteve 300 \
-savechkpteve3 1000 \
-epo 30 \
-epo3 20 \
-best1 data/best1 \
-best3 data/best3 \
-lr 4e-4
# -start log/base_0503