#!/usr/bin/env bash
python sct.py \
-m prefeats \
-cuda 0 \
-imgrt data/images \
-attsize 14 \
-incapjson data/dataset_coco.json \
-outfeatdir data/features_14 \
