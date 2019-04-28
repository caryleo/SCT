#!/usr/bin/env bash
python sct.py \
-m prefeats \
-cuda 3 \
-imgrt data/images/flickr30k \
-attsize 7 \
-incapjson data/dataset_flickr30k.json \
-outfeatdir data/features_30k \
