#!/usr/bin/env bash
python sct.py \
-m precaps \
-incapjson data/dataset_flickr30k.json \
-imgrt data/images/flickr30k \
-cuda 3 \
-outcapjson data/sct_caps2idx_30k.json \
-outcaph5 data/sct_caps_30k.h5
