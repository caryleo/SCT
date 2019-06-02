import json

file = 'data/sct_flickr30k_10.json'
data = json.load(open(file, 'r'))

list_few_coco = data['few']
count = 0
for index in range(len(data['images'])):
    image = data['images'][index]
    if image['split'] == 'test':
        if index in list_few_coco:
            count += 1

print(file)
print(count)
