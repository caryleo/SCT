import random
import os

outs = list()

for i in range(50):
    is_out = False
    while not is_out:
        out = random.randint(0, 600000)
        if out in outs:
            while out in outs:
                out = random.randint(0, 600000)
        outs.append(out)
        out_str = str(out).zfill(12)
        out_str = 'COCO_val2014_' + out_str + '.jpg'
        out_str = 'data/images/val2014/' + out_str
        is_out = os.path.isfile(out_str)
    print(i, out_str, is_out)
    cmd = 'cp ' + out_str + ' test_images/'
    os.system(cmd)
