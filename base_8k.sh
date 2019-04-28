#!/usr/bin/env bash
python sct.py \
-m train \
-id base_0425_8k \
-model base \
-cuda 3 \
-injson data/sct_caps2idx_8k.json \
-infeatdir data/features_8k_14 \
-incaph5 data/sct_caps_8k.h5 \
-outmemdir data/memory \
-inmemh5 data/memory/memory_base_0425_8k.h5 \
-tm 2 \
-tdir log/base_0425_8k \
-chkpt log/base_0425_8k \
-laneval 1 \
-batch 128 \
-batch3 32 \
-savechkpteve 500 \
-savechkpteve3 1000 \
-epo 20 \
-start log/base_0425_8k