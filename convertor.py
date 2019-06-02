import os
import json

annots_raw = json.load(open('data/dataset_flickr8k.json', 'r'))

annots = annots_raw['images'] # a list

data = {}
data['images'] = []
data['annotations'] = []
data['type'] = 'captions'
data['info'] = 'this is flickr30k info'
data['licenses'] = 'license of flickr30k'
snap = 1
for i in range(len(annots)):
    for sent in annots[i]['sentences']:
        images = {}
        annotations = {}

        images['file_name'] = annots[i]['filename']
        images['id'] = annots[i]['imgid']
        data['images'].append(images)

        annotations['image_id'] = annots[i]['imgid']
        annotations['caption'] = sent['raw']
        annotations['id'] = sent['sentid']
        data['annotations'].append(annotations)
    snap = snap + 1
    if snap % 100 == 0:
        print('processed %d images'%snap)


json.dump(data, open('coco-caption/annotations/captions_flickr8k.json', 'w'))