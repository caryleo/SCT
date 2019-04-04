"""
FILENAME:       SCT
DESCRIPTION:    the core
"""
import json
import logging
import torch

from tool import preprocess
from misc import options
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
        filemode='a'
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

    # DEBUG: opts
    para = vars(opts)
    logging.debug("Options input:\n" + json.dumps(para, indent=2))

    # cuda device
    device = torch.device("cuda:" + opts.cuda_device)
    logging.info("Device Using: %s " % device.__str__())

    # check mode
    if opts.mode == 'train':
        logging.info("Current core mode: Training")
        train.train(opts, device)
    elif opts.mode == 'eval':
        logging.info("Current core mode: Evaluating")
        eval.evaluation(opts, device)
    elif opts.mode == 'precaps':
        logging.info("Current core mode: Preprocessing captions")
        preprocess.preprocess_captions(opts)
    elif opts.mode == 'prefeats':
        logging.info("Current core mode: Preprocessing features")
        preprocess.preprocess_features(opts, device)
