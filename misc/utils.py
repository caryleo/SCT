import torch
import torch.nn as nn


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.shape
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


# make the tensor contiguous
def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


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


# set the learning rate for the optimizer
def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


# clip the gradient
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)
