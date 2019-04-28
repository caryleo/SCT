#!/usr/bin/env bash
python sct.py \
-m train \
-id base_0428 \
-model base \
-cuda 4,5,6,7 \
-injson data/sct_caps2idx.json \
-infeatdir data/features_14 \
-incaph5 data/sct_caps.h5 \
-outmemdir data/memory \
-inmemh5 data/memory/memory_base_0428.h5 \
-tm 1 \
-tdir log/base_0428 \
-chkpt log/base_0428 \
-laneval 1 \
-batch 256 \
-batch3 64 \
-savechkpteve 300 \
-savechkpteve3 1000 \
-epo 20 \
-best1 data/best1 \
-best3 data/best3
# -start log/base_0428 \