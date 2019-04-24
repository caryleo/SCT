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
        output = torch.sum(output) / torch.sum(mask) # 这里最后取得是batch*length级别的平均值

        return output


# relation classification loss
class RelationClassificationCriterion(nn.Module):
    def __init__(self, opts):
        super(RelationClassificationCriterion, self).__init__()
        self.opts = opts
        self.vocab_size = opts.vocabulary_size  # 9486 词汇表
        self.nouns_size = opts.nouns_size  # 7668 名词表
        self.vocabulary = opts.vocabulary  # 词汇索引到单词
        self.nouns = opts.nouns  # 名词单词到索引

    def forward(self, rel_ress, targets, mask):
        crit = nn.MSELoss()

        targets = targets[:, : rel_ress.shape[1]]
        mask = mask[:, : rel_ress.shape[1]]

        # 注意这里：和输出部分一样，这里没有针对BOS做处理，需要额外特殊处理一下，
        rel_ress = to_contiguous(rel_ress).view(-1, self.nouns_size) # (batch * length) * nouns
        targets = to_contiguous(targets).view(-1, 1) # (batch * length) * 1
        mask = to_contiguous(mask).view(-1, 1) # (batch * length) * 1
        rel_zero = torch.zeros(rel_ress.size(0), 1).cuda()
        rel = torch.cat((rel_zero, rel_ress), 1)

        targets_nouns_mask = targets.le(self.nouns_size).float().cuda() # 判断那些是名词 (batch * length) * 1
        targets = targets.float() * targets_nouns_mask
        targets_onehot = torch.zeros(rel.shape[0], rel.shape[1]).cuda()

        # print(rel_ress.size())
        # print(targets.size())
        # print(rel.size())
        # print(targets_nouns_mask.size())
        # print(targets_onehot.size())

        targets_onehot.scatter_(1, targets.long(), targets_nouns_mask) # 制造对应类别的onehot，对与非名词，直接zerohot
        return crit(rel, targets_onehot)


# stage 3 fuse loss
class FusionCriterion(nn.Module):
    def __init__(self, opts):
        super(FusionCriterion, self).__init__()
        self.LM = LanguageModelCriterion()
        self.RC = RelationClassificationCriterion(opts)

    def forward(self, inputs, rel_ress, targets, mask):
        output_LM = self.LM(inputs, targets, mask)
        output_RC = self.RC(rel_ress, targets, mask)
        return output_LM + output_RC
