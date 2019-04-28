#!/usr/bin/env bash
python sct.py \
-m train \
-id base_0425_30k \
-model base \
-cuda 3 \
-injson data/sct_caps2idx_30k.json \
-infeatdir data/features_30k_14 \
-incaph5 data/sct_caps_30k.h5 \
-outmemdir data/memory \
-inmemh5 data/memory/memory_base_0425_30k.h5 \
-tm 3 \
-tdir log/base_0425_30k \
-chkpt log/base_0425_30k \
-laneval 1 \
-batch 128 \
-batch3 16 \
-savechkpteve 500 \
-savechkpteve3 1000 \
-epo 20 \
-start log/base_0425_30k