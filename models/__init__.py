# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
FILENAME:       __init__.py
BY:             Gary 2019.3.12
LAST MODIFIED:  2019.3.12
DESCRIPTION:    load the model
"""

import os
import logging

from .ShowTellModel import ShowTellModel
from .FCModel import FCModel
from .OldModel import ShowAttendTellModel, AllImgModel
from .Att2inModel import Att2inModel
from .AttModel import *
from .ELKAModel import ELKAModel
from .RibekaModel import RibekaModel


def setup(opts):
    logging.info("Loading model: %s" % opts.caption_model)
    if opts.caption_model == 'show_tell':
        model = ShowTellModel(opts)
    elif opts.caption_model == 'show_attend_tell':
        model = ShowAttendTellModel(opts)
    # img is concatenated with word embedding at every time step as the input of lstm
    elif opts.caption_model == 'all_img':
        model = AllImgModel(opts)
    # FC model in self-critical
    elif opts.caption_model == 'fc':
        model = FCModel(opts)
    # Att2in model in self-critical
    elif opts.caption_model == 'att2in':
        model = Att2inModel(opts)
    # Att2in model with two-layer MLP img embedding and word embedding
    elif opts.caption_model == 'att2in2':
        model = Att2in2Model(opts)
    # Adaptive Attention model from Knowing when to look
    elif opts.caption_model == 'adaatt':
        model = AdaAttModel(opts)
    # Adaptive Attention with maxout lstm
    elif opts.caption_model == 'adaattmo':
        model = AdaAttMOModel(opts)
    # Top-down attention model
    elif opts.caption_model == 'tdbu':
        model = TopDownModel(opts)
    # My model
    elif opts.caption_model == 'elka':
        model = ELKAModel(opts)
    elif opts.caption_model == 'ribeka':
        model = RibekaModel(opts)
    else:
        raise Exception("Caption model not supported: {}".format(opts.caption_model))

    logging.info("Load model complete")

    # check compatibility if training is continued from previously saved model

    if vars(opts).get('start_from', None) is not None:
        # check if all necessary files exist
        logging.info("Load parameters from info.pkl")
        assert os.path.isdir(opts.start_from), "%s must be a a path" % opts.start_from
        assert os.path.isfile(os.path.join(opts.start_from, "info_" + opts.train_id + ".pkl")),\
            "info.pkl file does not exist in path %s" % opts.start_from
        model.load_state_dict(torch.load(os.path.join(opts.start_from, 'model.pth')))

    return model
