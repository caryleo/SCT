"""
FILENAME:       MODELS/__INIT__
DESCRIPTION:    setup for models
"""

import os
import logging

import torch

from .BaseModel import BaseModel
from .FinetuneModel import FinetuneModel
from .RelationModel import RelationModel
from .OutputModel import OutputModel


def setup(opts, stage_id=0):
    logging.info("Preparing model: %s" % opts.caption_model)
    if opts.caption_model == "base":
        model = BaseModel(opts)
    elif opts.caption_model == "finetune":
        model = FinetuneModel(opts)
    elif opts.caption_model == "relation":
        model = RelationModel(opts)
    elif opts.caption_model == "output":
        model = OutputModel(opts)
    else:
        raise Exception("Caption model not supported: {}".format(opts.caption_model))

    logging.info("Prepare model complete")

    # check compatibility if training is continued from previously saved model

    if vars(opts).get('start_from', None) is not None:
        # check if all necessary files exist
        logging.info("Resuming from existing data, Load parameters from info.pkl")
        assert os.path.isdir(opts.start_from), "%s must be a a path" % opts.start_from
        assert os.path.isfile(os.path.join(opts.start_from, "info_" + opts.train_id + ".pkl")),\
            "info.pkl file does not exist in path %s" % opts.start_from
        model.load_state_dict(torch.load(os.path.join(opts.start_from, 'model.pth')))
        return model
    elif stage_id == 2 or stage_id == 3:
        # 如果是指明了stage，没有指明start from，就加载最佳模型
        logging.warning("Loading parameters from info-best.pkl")
        assert os.path.isdir(opts.train_directory), "%s must be a directory" % opts.train_directory
        assert os.path.isfile(os.path.join(opts.train_directory, "info_" + opts.train_id + "-best.pkl")),\
            "no info-best.pkl in the directory"
        model.load_state_dict(torch.load(os.path.join(opts.train_directory, 'model-best.pth')))
        return model
    else:
        return model
