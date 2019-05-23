"""
FILENAME:       SCT
DESCRIPTION:    the program core
"""

import json
import logging
import torch
import os

from tool import preprocess
from utils import options
import train
import eval

if __name__ == "__main__":
    # arguments
    opts = options.parse_arg()

    # logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s**%(levelname)s\t%(message)s',
        datefmt='%Y.%m.%d-%H:%M:%S',
        filename='sct.log',
        filemode='w'
    )
    console = logging.StreamHandler()
    if opts.debug:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter('%(asctime)s**%(levelname)s\t%(message)s',
                          datefmt='%Y.%m.%d-%H:%M:%S'))
    logging.getLogger('').addHandler(console)

    # opts
    para = vars(opts)
    logging.debug("Options input: \n" + json.dumps(para, indent=2))

    # cuda device
    assert opts.cuda_device != '', "NO CUDA DEVICE SPECIFIED"

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.cuda_device
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    logging.info("Device Using: %s" % opts.cuda_device)

    # check mode
    if opts.mode == 'train':
        logging.info("Current core mode: Training")
        train.train(opts)
    elif opts.mode == 'eval':
        logging.info("Current core mode: Evaluating")
        eval.evaluation(opts)
    elif opts.mode == 'precaps':
        logging.info("Current core mode: Preprocessing captions")
        preprocess.preprocess_captions(opts)
    elif opts.mode == 'prefeats':
        logging.info("Current core mode: Preprocessing features")
        preprocess.preprocess_features(opts)
