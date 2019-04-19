"""
FILENAME:       DATALOADER
DESCRIPTION:    load data from preprocessed feature files
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

    # 重置迭代器，从头新建一个取数据子线程
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
        return self.nouns

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
        # self.noun_to_index = self.input_info_json["noun_to_index"]
        self.nouns = self.input_info_json["nouns"]
        # self.nouns_indices = self.input_info_json["nouns_indices"]
        self.dict_nouns = self.input_info_json["nouns_in_captions"]
        self.dict_nouns_captions = self.input_info_json["captions_for_nouns"]
        self.vocabulary_size = len(self.index_to_word)
        self.nouns_size = len(self.nouns)
        logging.info('Size of vocabulary: %d' % self.vocabulary_size)
        logging.info('Size of nouns: %d', self.nouns_size)
        logging.info('Load input json file complete')

        # load the captions h5 with memory mapping
        logging.info('Loading input captions h5 file: %s' % opts.input_captions_h5)
        self.captions_h5 = h5py.File(self.opts.input_captions_h5, 'r', driver='core')
        captions_size = self.captions_h5['captions'].shape
        self.max_caption_length = captions_size[1]
        logging.debug("Maximal Caption length: %d" % self.max_caption_length)
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

        # 针对三个split分别创建一个取数据的进程
        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split,
                                                        self,
                                                        split == 'train')

        # Terminate the child process when the parent exists 在主进程退出时，终止所有的子进程
        def cleanup():
            # logging.info('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)

    def get_batch(self, split, batch_size=None, caps_per_img=None):
        batch_size = batch_size or self.batch_size
        caps_per_img = caps_per_img or self.captions_per_image

        fc_batch = list()
        att_batch = list()
        # 注意这里，句子长度多了两位
        caption_batch = np.zeros(
            [batch_size * caps_per_img, self.max_caption_length + 2], dtype='int')
        mask_batch = np.zeros(
            [batch_size * caps_per_img, self.max_caption_length + 2], dtype='float32')

        # 这个变量用于判断一次是否已经到头
        wrapped = False

        info = list()
        gts = list()

        # 每一次返回，batch_size次调用
        for i in range(batch_size):
            # fetch image 取一次是取一个图片的fc和att特征， 为了方便对齐，直接复制5份
            feat_fc, feat_att, ix, tmp_wrapped = self._prefetch_process[split].get()
            fc_batch += [feat_fc] * caps_per_img
            att_batch += [feat_att] * caps_per_img

            # fetch the sequence labels 因为索引定位是从1开始的，需要减1
            index1 = self.index_start[ix] - 1  # label_start_ix starts from 1
            index2 = self.index_end[ix] - 1
            num_cap = index2 - index1 + 1  # number of captions available for this image

            assert num_cap > 0, 'an image does not have any caption.'

            if num_cap < caps_per_img:
                # we need to subsample (with replacement), 如果实际对应的描述量不够制定的描述量，需要采样补足
                caps = np.zeros([caps_per_img, self.max_caption_length], dtype='int')
                for q in range(caps_per_img):
                    ixl = random.randint(index1, index2)
                    caps[q, :] = self.captions_h5['captions'][ixl, :self.max_caption_length]
            else:
                # 如果多了，就取够就好
                ixl = random.randint(index1, index2 - caps_per_img + 1)
                caps = self.captions_h5['captions'][ixl: ixl + caps_per_img, :self.max_caption_length]

            # put the caption in the middle: [0] is 0 and [max_length+1] is 0 把整个描述放到中间，前后补0，实际上充当BOS和EOS
            caption_batch[i * caps_per_img: (i + 1) * caps_per_img, 1: self.max_caption_length + 1] = caps

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation, indices are 1-indexed 取出参考描述
            gts.append(
                self.captions_h5['captions'][self.index_start[ix] - 1:
                                             self.index_end[ix]])

            # record associated info as well 信息块保存的 image的索引，图片的COCOID，以及路径
            info_dict = {'index': ix,
                         'id': self.input_info_json['images'][ix]['cocoid'],
                         'file_path': self.input_info_json['images'][ix]['filepath']}
            info.append(info_dict)

        # generate mask 生成掩模，明确有意义的位（代替长度记录，更方便一下额）
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, caption_batch)))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        # 一次batch的全部数据如下：fc特征扩展batch，att特征扩展batch，描述batch，从参考描述batch，掩模batch，当前split的迭代边界，上面有关image的信息块
        data_all = {'fc_feats': np.stack(fc_batch),
                    'att_feats': np.stack(att_batch),
                    'captions': caption_batch,
                    'gts': gts,
                    'masks': mask_batch,
                    # bounds用于明确边界
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


# 子集随机采样，因为需要对整个数据库重新划分一下，因此是取的索引的子集，这里直接把随机采样的定义为序列采样（可以考虑修改一下）
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
        self.if_shuffle = if_shuffle  # 根据源代码，只对train集做shuffle

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
        # sampler 从当前split的当前iterator开始采样， 这里batchsize是1，因为他v把构造batch的过程放到了loader里面
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
            wrapped = True  # epoch结束的标记
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()  # 没有split_loader，新建一个

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()  # 新的epoch，新建一个
        assert tmp[2] == ix, "index not equal"

        return tmp + [wrapped]
