# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
FILENAME:       dataloader.py
BY:             Gary 2019.3.16
LAST MODIFIED:  2019.3.16
DESCRIPTION:    load the data for training and evaluating
"""

import json
import h5py
import os
import numpy as np
import random
import torch.utils.data as data
import logging
import multiprocessing


class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split,
                                                    self, split == 'train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocabulary_size

    def get_vocab(self):
        return self.index_to_word

    def get_nouns_size(self):
        return self.nouns_size

    def get_nouns(self):
        return self.index_to_noun

    def get_seq_length(self):
        return self.max_caption_length

    def read_files(self):
        self.feats_fc = h5py.File(os.path.join(
            self.opts.input_features_directory, 'feats_fc.h5'), 'r')
        self.feats_att = h5py.File(os.path.join(
            self.opts.input_features_directory, 'feats_att.h5'), 'r')

    def get_data(self, ix):
        self.read_files()
        index = str(self.input_info_json['images'][ix]['cocoid'])
        return (np.array(self.feats_fc[index]).astype('float32'),
                np.array(self.feats_att[index]).astype('float32'), ix)

    def __init__(self, opts):
        self.opts = opts

        # info for data loading
        self.batch_size = self.opts.batch_size
        self.captions_per_image = opts.captions_per_image
        # self.use_att = getattr(opts, 'use_att', True)

        # load json file which contains additional information about dataset
        logging.info('Loading input json file: %s' % opts.input_json)
        self.input_info_json = json.load(open(self.opts.input_json))
        self.index_to_word = self.input_info_json['index_to_word']
        self.index_to_noun = self.input_info_json["index_to_noun"]
        self.dict_nouns = self.input_info_json["nouns_in_capions"]
        self.dict_nouns_captions = self.input_info_json["captions_for_nouns"]
        self.vocabulary_size = len(self.index_to_word)
        self.nouns_size = len(self.index_to_noun)
        logging.info('Size of vocabulary: %d' % self.vocabulary_size)
        logging.info('Size of noun: %d', self.nouns_size)
        logging.info('Load input json file complete')

        # load the captions h5 with memory mapping
        logging.info('Loading input captions h5 file: %s' % opts.input_captions_h5)
        self.captions_h5 = h5py.File(self.opts.input_captions_h5, 'r', driver='core')
        captions_size = self.captions_h5['captions'].shape
        self.max_caption_length = captions_size[1]
        logging.debug("Caption length: %d" % self.max_caption_length)
        self.index_start = self.captions_h5['index_start'][:]
        self.index_end = self.captions_h5['index_end'][:]
        self.caption_length = self.captions_h5['caption_lengths'][:]
        self.num_images = self.index_start.shape[0]
        logging.info('Load input captions h5 file complete')

        # separate out indexes for each of the provided splits
        logging.info("Spliting into 3 datasets")
        self.split_index = {'train': list(), 'val': list(), 'test': list()}
        for index in range(len(self.input_info_json['images'])):
            image = self.input_info_json['images'][index]
            if image['split'] == 'train':
                self.split_index['train'].append(index)
            elif image['split'] == 'val':
                self.split_index['val'].append(index)
            elif image['split'] == 'test':
                self.split_index['test'].append(index)
            elif opts.train_only == 0:  # restval split
                self.split_index['train'].append(index)

        logging.info('assigned %d images to split train' % len(self.split_index['train']))
        logging.info('assigned %d images to split val' % len(self.split_index['val']))
        logging.info('assigned %d images to split test' % len(self.split_index['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        # 这部分是预取特征的操作，目前看不懂
        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split,
                                                        self,
                                                        split == 'train')

        # Terminate the child process when the parent exists

        def cleanup():
            logging.info('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)

    def get_batch(self, split, batch_size=None, caps_per_img=None):
        batch_size = batch_size or self.batch_size
        caps_per_img = caps_per_img or self.captions_per_image

        fc_batch = list()
        att_batch = list()
        caption_batch = np.zeros(
            [batch_size * caps_per_img, self.max_caption_length + 2], dtype='int')
        mask_batch = np.zeros(
            [batch_size * caps_per_img, self.max_caption_length + 2], dtype='float32')

        wrapped = False

        info = list()
        gts = list()

        for i in range(batch_size):
            # fetch image
            feat_fc, feat_att, ix, tmp_wrapped = self._prefetch_process[split].get()
            fc_batch += [feat_fc] * caps_per_img
            att_batch += [feat_att] * caps_per_img

            # fetch the sequence labels
            index1 = self.index_start[ix] - 1  # label_start_ix starts from 1
            index2 = self.index_end[ix] - 1
            num_cap = index2 - index1 + 1  # number of captions available for this image

            assert num_cap > 0, 'an image does not have any caption.'

            if num_cap < caps_per_img:
                # we need to subsample (with replacement)
                caps = np.zeros([caps_per_img, self.max_caption_length], dtype='int')
                for q in range(caps_per_img):
                    ixl = random.randint(index1, index2)
                    caps[q, :] = self.captions_h5['captions'][ixl, :self.max_caption_length]
            else:
                ixl = random.randint(index1, index2 - caps_per_img + 1)
                caps = self.captions_h5['captions'][ixl: ixl + caps_per_img, :self.max_caption_length]
            # put the caption in the middle: [0] is 0 and [max_length+1] is 0
            caption_batch[i * caps_per_img: (i + 1) * caps_per_img, 1: self.max_caption_length + 1] = caps

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation, indices are 1-indexed
            gts.append(
                self.captions_h5['captions'][self.index_start[ix] - 1:
                                             self.index_end[ix]])

            # record associated info as well
            info_dict = {'index': ix,
                         'id': self.input_info_json['images'][ix]['cocoid'],
                         'file_path': self.input_info_json['images'][ix]['filepath']}
            info.append(info_dict)

        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, caption_batch)))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        data_all = {'fc_feats': np.stack(fc_batch),
                    'att_feats': np.stack(att_batch),
                    'captions': caption_batch,
                    'gts': gts,
                    'masks': mask_batch,
                    'bounds': {'it_pos_now': self.iterators[split],
                               'it_max': len(self.split_index[split]),
                               'wrapped': wrapped},
                    'info': info}

        return data_all

    # It's not coherent to make DataLoader a subclass of Dataset,
    # but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according
    # the index. However, it's minimum change to switch to pytorch data loading
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        return self.get_data(ix)

    def __len__(self):
        return len(self.input_info_json['images'])


# 后面的部分看不懂 =============================================================================
class ArraySampler(data.sampler.SubsetRandomSampler):
    def __iter__(self):
        return iter(self.indices)


class BlobFetcher:
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name,
        caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases:
        1. not hasattr(self, 'split_loader'): Resume from previous training.
        Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in
         the get_minibatch_inds already.
        """
        # batch_size is 0, the merge is done in DataLoader class
        sampler = ArraySampler(
            self.dataloader.split_index[self.split][self.dataloader.iterators[self.split]:])
        self.split_loader = iter(
            data.DataLoader(dataset=self.dataloader,
                            batch_size=1,
                            sampler=sampler,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=multiprocessing.cpu_count(),
                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_index[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_index[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_index[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()
        assert tmp[2] == ix, "index not equal"

        return tmp + [wrapped]
