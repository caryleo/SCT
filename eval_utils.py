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
    if dataset == 'coco':
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif dataset == 'flickr8k':
        annFile = 'coco-caption/annotations/captions_flickr8k.json'
    else:
        annFile = 'coco-caption/annotations/captions_flickr30k.json'

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


def classification_eval(loader, outputs, nouns_in_caption):
    # print(len(outputs))
    # print(len(nouns_in_caption))
    assert len(outputs) == len(nouns_in_caption), "OOPSIE!"

    length = len(loader.nouns)
    values = np.zeros((length + 1, 3)) # tp, fp, np
    # print(values.shape)
    for i in range(len(outputs)):
        output = outputs[i]
        nouns_single = nouns_in_caption[i]
        for noun in nouns_single:
            if noun > 0:
                if loader.get_vocab()[str(noun)] == "UNK": continue
            if noun not in output:
                values[noun, 2] += 1 # fn, true but not appears
            else:
                values[noun, 0] += 1 # tp, nouns and true

        other = list()
        for word in output:
            if word > 0:
                if loader.get_vocab()[str(word)] == "UNK": continue
            if word <= length and word not in nouns_single and word not in other:
                values[word, 1] += 1 # fp, nouns and false

    tp = values[:, 0]
    fp = values[:, 1]
    fn = values[:, 2]

    precision = np.zeros((length + 1, 1))
    recall = np.zeros((length + 1, 1))
    F1 = np.zeros((length + 1, 1))

    for i in range(length + 1):
        if tp[i] + fp[i] == 0:
            precision[i] = 0.0
        else:
            precision[i] = float(tp[i]) / (tp[i] + fp[i])

        if tp[i] + fn[i] == 0:
            recall[i] = 0.0
        else:
            recall[i] = float(tp[i]) / (tp[i] + fn[i])

        if precision[i] + recall[i] == 0:
            F1[i] = 0.0
        else:
            F1[i] = 2.0 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    return (precision, recall, F1)


def eval_split(model, crit, loader, eval_kwargs={}):
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    device = eval_kwargs.get('cuda_device', -1)
    logging.info("Evaluating by device: %s" % device)
    logging.info("Dataset used: %s" % dataset)
    stage_id = eval_kwargs.get('stage', 0)
    metric = eval_kwargs.get('metric', 1)

    # Make sure in the evaluation mode
    model.eval()
    # 每一次eval都重置一下 ！！！！
    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = list()
    predictions_index = list()
    nouns_in_captions = list()

    logging.info("Sampling for evaluation")
    while True:
        nouns_in_caption = None
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('captions', None) is not None:
            # forward the model to get loss 这一部分直接出loss
            tmp = [data['fc_feats'], data['att_feats'], data['captions'], data['masks']]
            if metric == 4:
                nouns_in_caption = data['nouns_in_caption']

            with torch.no_grad():
                tmp = [torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks = tmp
                outputs, rel_ress = model(fc_feats, att_feats, labels, stage_id)
                if stage_id == 1 or stage_id == 2:  #
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
            tmp = [torch.from_numpy(_).cuda() for _ in tmp]
            fc_feats, att_feats = tmp
            # forward the model to also get generated samples for each image 取样评估整体性能
            seq, _, _ = model.module.sample(fc_feats, att_feats, stage_id, eval_kwargs)

        seq = seq.to("cpu").numpy()
        if metric == 4:
            predictions_index.extend(seq)

        # print(len(predictions_index))
        if metric == 4:
            tag = 0
            nouns_in_caption_single = list()
            while tag < len(nouns_in_caption):
                single = list()
                for _ in range(loader.captions_per_image):
                    entry = nouns_in_caption[tag] # a list of tuple (word index, pos in caption)
                    for noun_pair in entry: # each tuple
                        noun = noun_pair[0]
                        if noun not in single:
                            single.append(noun) # put all distinct nouns in single for each image
                    tag += 1
                nouns_in_caption_single.append(single) # for the batch
            nouns_in_captions.extend(nouns_in_caption_single)

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

            if metric == 4:
                predictions_index.pop()
                nouns_in_captions.pop()

        logging.debug('evaluating validation performance... %d / %d (%f)' % (ix0, ix1, loss))
        if ix0 % (loader.batch_size * 8) == 0:
            logging.info('evaluating validation performance... %d / %d (%f)' % (ix0, ix1, loss))

        if data['bounds']['wrapped']:
            break

        if 0 <= num_images <= n:
            break
    logging.info("Sampling complete")

    lang_stats = None

    results = None

    if metric == 1:
        logging.info("Language evaluation")
        logging.getLogger('').setLevel(logging.ERROR)

        if lang_eval == 1:
            lang_stats = language_eval(dataset, predictions, eval_kwargs['train_id'], split)
        logging.getLogger('').setLevel(logging.INFO)
        logging.info("Language evaluation complete")
    else:
        logging.info("F val evaluation")
        results = classification_eval(loader, predictions_index, nouns_in_captions)
        logging.info("F val evaluation complete")

    # Switch back to training mode
    model.train()
    if metric == 1:
        return loss_sum / loss_evals, predictions, lang_stats
    else:
        return loss_sum / loss_evals, predictions, results