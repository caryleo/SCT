import torch
from torch import nn as nn

from utils.utils import to_contiguous


# language model loss
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, inputs, targets, mask):
        # truncate to the same size
        targets = targets[:, :inputs.shape[1]]
        mask = mask[:, :inputs.shape[1]]
        inputs = to_contiguous(inputs).view(-1, inputs.shape[2])
        targets = to_contiguous(targets).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - inputs.gather(1, targets) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output