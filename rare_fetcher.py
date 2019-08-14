import json
import os
import random

data = json.load(open('data/sct_coco_50.json'))
file = json.load(open('data/dataset_coco.json'))

nouns = data['nouns']
i = 0
outs = list()

for image in file['images']:
    flag = False
    for sentence in image['sentences']:
        for token in sentence['tokens']:
            if token in nouns:
                flag = True
                break

        if flag:
            break
    if flag:
        #print(image['filepath'], image['filename'])
        outs.append(os.path.join('data/images', image['filepath'], image['filename']))
        i += 1

    if i % 1000 == 0:
        print('HA!', i)

len_out = len(outs)
print(len_out)

for i in range(20):
    index = random.randint(0, len_out)
    out = outs[index]
    print(i, out)
    cmd = 'cp ' + out + ' test_rare_images/'
    os.system(cmd)