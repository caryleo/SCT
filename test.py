import json
import csv

file_1 = 'eval_results/finetune_0515_F1_100.csv'
file_2 = 'data/sct_coco_100.json'
file_3 = 'eval_results/finetune_0515_F1-stage-1.json'
file_4 = 'eval_results/finetune_0515_F1-stage-3.json'

file = open(file_1, 'w')
json2 = json.load(open(file_2, 'r'))
json3 = json.load(open(file_3, 'r'))
json4 = json.load(open(file_4, 'r'))

vocabulary = json2['index_to_word']
nouns = json2['nouns']

csvfile = csv.writer(file, dialect='excel')

for noun in nouns:
    # word = vocabulary[noun]
    if noun in json3:
        values1 = json3[noun]
        values3 = json4[noun]

    precision1 = values1['precision']
    recall1 = values1['recall']
    F11 = values1['F1']

    precision3 = values3['precision']
    recall3 = values3['recall']
    F13 = values3['F1']

    inc_p = precision3 - precision1
    inc_r = recall3 - recall1
    inc_f = F13 - F11

    if precision3 > 0.0 or recall3 > 0.0 or F13 > 0.0:
        csvfile.writerow([noun, precision1, precision3, inc_p, recall1, recall3, inc_r, F11, F13, inc_f])