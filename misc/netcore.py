#! /usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
FILENAME:       netcore.py
BY:             @ruotianluo 2019.3.16
LAST MODIFIED:  2019.3.16
DESCRIPTION:    customized version of ResNet by @ruotianluo, for feature extraction
"""

import torch.nn as nn
import torch.nn.functional as F


class my_resnet(nn.Module):
    def __init__(self, resnet):
        super(my_resnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2).squeeze()  # 取fc特征，直接取均值到一个维度上
        conv = F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().permute(1, 2, 0)  # 取conv特征，取出来之后就将batch维度调整到第三维

        return fc, conv
