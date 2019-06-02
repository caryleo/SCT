import json
import os
import numpy as np
import torch
import skimage
import skimage.io
import logging
import torch.nn as nn

from torchvision import transforms as trn

preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from utils.featcore import my_resnet
import utils.resnet


class DataLoaderRaw():

    def __init__(self, opts):
        self.opt = opts
        self.coco_json = opts.get('coco_json', '')
        self.folder_path = opts.get('folder_path', '')

        self.batch_size = opts.get('batch_size', 1)
        self.captions_per_image = 1

        # Load resnet
        self.cnn_model = opts.get('cnn_model', 'resnet101')
        self.my_resnet = getattr(utils.resnet, self.cnn_model)()
        self.my_resnet.load_state_dict(torch.load('./data/models/' + self.cnn_model + '.pth'))
        self.my_resnet = nn.DataParallel(my_resnet(self.my_resnet))
        self.my_resnet.cuda()
        self.my_resnet.eval()

        # load the json file which contains additional information about the dataset
        logging.info('DataLoaderRaw loading images from folder: %s' % self.folder_path)

        self.files = []
        self.ids = []

        if len(self.coco_json) > 0:
            logging.info('reading from ' + opts.coco_json)
            # read in filenames from the coco-style json file
            self.coco_annotation = json.load(open(self.coco_json))
            for k, v in enumerate(self.coco_annotation['images']):
                fullpath = os.path.join(self.folder_path, v['file_name'])
                self.files.append(fullpath)
                self.ids.append(v['id'])
        else:
            # read in all the filenames from the folder
            logging.info('listing all images in directory ' + self.folder_path)

            def isImage(f):
                supportedExt = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM']
                for ext in supportedExt:
                    start_idx = f.rfind(ext)
                    if start_idx >= 0 and start_idx + len(ext) == len(f):
                        return True
                return False

            n = 1
            for root, dirs, files in os.walk(self.folder_path, topdown=False):
                for file in files:
                    fullpath = os.path.join(self.folder_path, file)
                    if isImage(fullpath):
                        self.files.append(fullpath)
                        self.ids.append(str(n))  # just order them sequentially
                        n = n + 1

        self.N = len(self.files)
        logging.info('DataLoaderRaw found %d images' % self.N)

        self.iterator = 0

    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size

        # pick an index of the datapoint to load next
        fc_batch = np.ndarray((batch_size, 2048), dtype='float32')
        att_batch = np.ndarray((batch_size, 14, 14, 2048), dtype='float32')
        max_index = self.N
        wrapped = False
        infos = []

        for i in range(batch_size):
            ri = self.iterator
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
                # wrap back around
            self.iterator = ri_next

            img = skimage.io.imread(self.files[ri])

            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate((img, img, img), axis=2)

            img = img[:, :, :3].astype('float32') / 255.0
            img = torch.from_numpy(img.transpose([2, 0, 1])).cuda()
            img = preprocess(img)
            with torch.no_grad():
                img = img.unsqueeze(0)
                tmp_fc, tmp_att = self.my_resnet(img)

            fc_batch[i] = tmp_fc.data.cpu().float().numpy()
            att_batch[i] = tmp_att.data.cpu().float().numpy()

            info_struct = {}
            info_struct['id'] = self.ids[ri]
            info_struct['file_path'] = self.files[ri]
            infos.append(info_struct)

        data = {}
        data['fc_feats'] = fc_batch
        data['att_feats'] = att_batch.reshape(batch_size, -1, 2048)
        data['att_masks'] = None
        data['bounds'] = {'it_pos_now': self.iterator, 'it_max': self.N, 'wrapped': wrapped}
        data['info'] = infos

        return data

    def reset_iterator(self, split):
        self.iterator = 0

    def get_vocab_size(self):
        return len(self.index_to_word)

    def get_vocab(self):
        return self.index_to_word
