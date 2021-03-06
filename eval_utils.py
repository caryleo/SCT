import torch
import logging

import numpy as np
import json
from json import encoder
import os
import utils.misc as utils


def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    device = eval_kwargs.get('device', torch.device("cuda:" + str(eval_kwargs.get('cuda_device', 0))))
    logging.info("Evaluating by device: %s", device)
    stage_id = eval_kwargs.get('stage', 0)

    # Make sure in the evaluation mode
    model.eval()
    # 每一次eval都重置一下 ！！！！
    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('captions', None) is not None:
            # forward the model to get loss 这一部分直接出loss
            tmp = [data['fc_feats'], data['att_feats'], data['captions'], data['masks']]
            with torch.no_grad():
                tmp = [torch.from_numpy(_).to(device=device) for _ in tmp]
                fc_feats, att_feats, labels, masks = tmp
                outputs, rel_ress = model(fc_feats, att_feats, labels, stage_id)
                if stage_id == 1 or stage_id == 2: #
                    loss = crit(outputs, labels[:, 1:], masks[:, 1:]).item()
                elif stage_id == 3:
                    loss = crit(outputs, rel_ress, labels[:, 1:], masks[:, 1:]).item()
                else:
                    logging.error("INCORRECT STAGE ID!!!")

            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample 为啥要这么写呀？？？
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.captions_per_image],
               data['att_feats'][np.arange(loader.batch_size) * loader.captions_per_image]]
        with torch.no_grad():
            tmp = [torch.from_numpy(_).to(device=device) for _ in tmp]
            fc_feats, att_feats = tmp
            # forward the model to also get generated samples for each image 取样评估整体性能
            seq, _, _ = model.sample(fc_feats, att_feats, stage_id, eval_kwargs)

        seq = seq.to("cpu").numpy()

        # set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['info'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['info'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'],
                                            data['info'][k]['file_path']) + '" vis/imgs/img' + str(
                    len(predictions)) + '.jpg'  # bit gross
                logging.info("Executing: %s" % cmd)
                os.system(cmd)

            logging.debug('image %s: %s' % (entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        if num_images != -1:
            ix1 = min(ix1, num_images)

        for i in range(n - ix1):
            predictions.pop()

        logging.debug('evaluating validation performance... %d / %d (%f)' % (ix0 - 1, ix1, loss))
        if (ix0 - 1) % (loader.batch_size * 10) == 0 :
            logging.info('evaluating validation performance... %d / %d (%f)' % (ix0 - 1, ix1, loss))
        # if (ix0 - 1) % (loader.batch_size * 10):
        #     logging.debug('evaluating validation performance... %d / %d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break

        if 0 <= num_images <= n:
            break

    logging.getLogger('').setLevel(logging.ERROR)
    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['train_id'], split)
    logging.getLogger('').setLevel(logging.INFO)
    # Switch back to training mode
    model.train()
    return loss_sum / loss_evals, predictions, lang_stats
