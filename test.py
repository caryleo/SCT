import argparse
import os
import torch

parser = argparse.ArgumentParser()

parser.add_argument('-t', type=str)

opts = parser.parse_args()

print(vars(opts))

os.environ['CUDA_VISIBLE_DEVICES'] = opts.t
a = torch.Tensor(10, 5).random_() % 10
a = a.cuda()
