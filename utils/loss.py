"""
FILENAME:       LOSS
DESCRIPTION:    loss functions for the model
"""

import torch
from torch import nn as nn

from utils.misc import to_contiguous


# language model loss
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, inputs, targets, mask):
        # truncate to the same size (因为生成的内容可能恰好短一点)
        targets = targets[:, :inputs.shape[1]] # batch * length
        mask = mask[:, :inputs.shape[1]] # batch * length
        inputs = to_contiguous(inputs).view(-1, inputs.shape[2]) # (batch * length) * (vocab + 1)
        targets = to_contiguous(targets).view(-1, 1) # (batch * length) * 1
        mask = to_contiguous(mask).view(-1, 1) # (batch * length) * 1
        output = - inputs.gather(1, targets) * mask # 取出特定位置的词的概率，mask用于保证特定时刻是否有词 # (batch * length) * 1
        output = torch.sum(output) / torch.sum(mask)

        return output


# relation classification loss
class RelationClassificationCriterion(nn.Module):
    def __init__(self, opts):
        super(RelationClassificationCriterion, self).__init__()

    def forward(self, inputs, targets, mask):
        pass


# stage 3 fuse loss
class FusionCriterion(nn.Module):
    def __init__(self, opts):
        super(FusionCriterion, self).__init__()
        self.LM = LanguageModelCriterion()
        self.RC = RelationClassificationCriterion(opts)

    def forward(self, inputs, targets, mask):
        output_LM = self.LM(inputs, targets, mask)
        output_RC = self.RC(inputs, targets, mask)
        return output_LM + output_RC
