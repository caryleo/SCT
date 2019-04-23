"""
FILENAME:       MODELS/BaseModel
DESCRIPTION:    BaseModel for SCT
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseModel(nn.Module):
    def __init__(self, opts):
        super(BaseModel, self).__init__()

        # self.rnn_type = opts.rnn_type
        self.opts = opts
        self.rnn_size = opts.rnn_size  # 512 rnn隐藏单元个数
        self.num_layers = opts.num_layers  # 1 LSTM层数，在这个模型里面没有意义
        self.drop_prob_lm = opts.dropout_prob  # dropout概率，暂时不管
        self.ss_prob = 0.0  # Schedule sampling probability SS概率，暂时不管
        self.max_caption_length = opts.max_caption_length  # 16 caption截断长度

        self.vocab_size = opts.vocabulary_size  # 9486 词汇表
        self.nouns_size = opts.nouns_size  # 7668 名词表
        self.vocabulary = opts.vocabulary  # 词汇索引到单词
        self.nouns = opts.nouns  # 名词单词到索引
        self.fuse_coefficient = opts.fuse_coefficient

        self.input_encoding_size = opts.input_encoding_size  # 512 单词输入内部特征表示维度
        self.fc_feat_size = opts.fc_feat_size  # 2048 FC特征长度，这个模型暂时用不到
        self.att_feat_size = opts.att_feat_size  # 2048 ATT特征长度，att*att*2048 14*14*2048
        self.att_hid_size = opts.att_hid_size  # 512 ATT过程中的中间维度（线性转换后到生成分数前）

        self.word_embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.drop_prob_lm))  # 9487 - 512 单词嵌入，多出一个是对应BOS
        # self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
        #                               nn.ReLU(),
        #                               nn.Dropout(self.drop_prob_lm)) # 2048 - 512 FC特征嵌入，这里用不上，因为Att2in2模型不需要FC特征

        self.fc_embed = lambda x: x  # 2048 - 2048 由于本模型不需要FC，因此这里做一个简单的处理（用于保持模型一致）

        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                       nn.ReLU(),
                                       nn.Dropout(self.drop_prob_lm))  # 512 - 512 att结果到hid的维度一致

        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)  # 512 - 9487 最后的输出映射

        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)  # 512 - 512 用于计算att向量的结果，提前计算节省内存

        self.core = BaseCore(opts)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        # h 和 c
        return (weight.new(self.num_layers, bsz, self.rnn_size).zero_(),
                weight.new(self.num_layers, bsz, self.rnn_size).zero_())

    def memory_ready(self):
        self.core.memory_ready()

    def memory_finish(self):
        return self.core.memory_finish()

    def set_memory(self, memory):
        self.core.set_memory(memory)

    def forward(self, fc_feats, att_feats, captions, stage_id):
        batch_size = fc_feats.size(0)  # 64 取batch的大小
        state = self.init_hidden(batch_size)

        outputs = list()  # 按时间点排序
        rel_ress = list()

        # embed fc and att feats 提前计算好节省时间和空间
        # att特征在rnn中的内部表示，用于形成注意力上下文
        fc_feats = self.fc_embed(fc_feats)  # fc特征映射一下，变成 batch * rnn_size 64*512
        _att_feats = self.att_embed(
            att_feats.view(-1, self.att_feat_size))  # 调整成二维好计算，(batch * att * att) * rnn_size (64*14*14)*512
        att_feats = _att_feats.view(
            *(att_feats.size()[:-1] + (self.rnn_size,)))  # 再调整回来，batch*att*att*rnn_size 64*14*14*512

        # Project the attention feats first to reduce memory and computation consumptions. 提前计算好节省时间和空间
        # att特征在att中的内部表示，用于形成注意力分数
        context_att_feats = self.ctx2att(
            att_feats.view(-1, self.rnn_size))  # 上下文转att结果，同理，(batch*att*att)*rnn_size (64*14*14)*512
        context_att_feats = context_att_feats.view(
            *(att_feats.size()[:-1] + (self.att_hid_size,)))  # 调整回合适的维度，同理，batch*att*att*att_hid_size 64*14*14*512

        # 取caption的长度，即最大长度16+2,减1是为了将最后一个EOS去掉
        for i in range(captions.size(1) - 1):

            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwise no need to sample
                # 这一部分ss的情况，暂时不管
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    wordt = captions[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    wordt = captions[:, i].data.clone()
                    # prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    # it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    prob_prev = torch.exp(outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)
                    wordt.index_copy_(0, sample_ind,
                                      torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    wordt.requires_grad = False
            else:
                # t=0 送入BOS，其他时刻送入参考caption对应时间点的字符
                wordt = captions[:, i].clone()
                # break if all the sequences end
            # print(captions.size())
            if i >= 1 and captions[:, i].data.sum() == 0:
                # 如果所有的caption这个时候都是0了，就结束了
                break

            xt = self.word_embed(wordt)  # 将单词编码，batch * encoding，64*9487 - 64*512

            output, state, rel_res = self.core(wordt, xt, fc_feats, att_feats, context_att_feats, state, stage_id)
            LMvocab = self.logit(output)
            if stage_id == 1 or stage_id == 2:
                # 只使用语言模型输出
                output = LMvocab
            else:
                # 使用线性融合
                # for i in range(batch_size):
                #     for j in range(self.vocab_size + 1):
                #         if self.vocabulary.get(output[i][j]) in self.nouns:
                #             index = self.nouns.get(self.vocabulary.get(output[i][j]))
                #             output[i][j] = self.fuse_coefficient * LMvocab[i][j] + \
                #                            (1 - self.fuse_coefficient) * rel_res[i][index]
                #         else:
                #             output[i][j] = LMvocab[i][j]
                rel_zero = LMvocab[:, 0].view(-1, 1)
                rel_others = LMvocab[:, self.nouns_size + 1:]  # barch * (vocab - nouns)
                # print(rel_res.size())
                # print(rel_zero.size())
                # print(rel_others.size())
                rel = torch.cat((rel_zero, rel_res, rel_others), 1)
                output = self.fuse_coefficient * LMvocab + (1 - self.fuse_coefficient) * rel
                rel_res = F.softmax(rel_res, 1)

            output = F.log_softmax(output, 1)  # 出来的结果做一下log和softmax，这一已经映射到了词汇表 batch * (vocab + 1)
            outputs.append(output) # length (batch * (vocab+1))
            rel_ress.append(rel_res)
        if stage_id == 1 or stage_id == 2:
            return torch.cat([_.unsqueeze(1) for _ in outputs], 1), None
        else:
            return torch.cat([_.unsqueeze(1) for _ in outputs], 1), torch.cat([_.unsqueeze(1) for _ in rel_ress], 1) # batch * length * (vocab + 1), batch * length * nouns

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state):
        # 'it' is Variable contraining a word index
        xt = self.word_embed(it)

        output, state = self.core(it, xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state)
        logprobs = F.log_softmax(self.logit(output))

        return logprobs, state

    def sample_beam(self, fc_feats, att_feats, stage_id, opts={}):
        beam_size = opts.get('beam_size', 64)  # batch
        batch_size = fc_feats.size(0)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.max_caption_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.max_caption_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k + 1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats = att_feats[k:k + 1].expand(*((beam_size,) + att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k + 1].expand(*((beam_size,) + p_att_feats.size()[1:])).contiguous()

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    it.requires_grad = False
                    xt = self.word_embed(it)

                output, state, rel_ress = self.core(it, xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state, stage_id)
                logprobs = F.log_softmax(self.logit(output))

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                  opts=opts)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, att_feats, stage_id, opts={}):
        sample_max = opts.get('sample_greedy', 1)
        beam_size = opts.get('beam_size', 1)
        temperature = opts.get('temperature', 1.0)
        if beam_size > 1: # 因为目前用不上beam search，因此这部分先不改了
            return self.sample_beam(fc_feats, att_feats, stage_id, opts)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        seq = list()
        seqLogprobs = list()
        rel_ress = list()
        for t in range(self.max_caption_length + 1):
            if t == 0:  # input <bos> # 采样时，使用fc的信息做参考生成一个batch_size的全零向量做BOS
                it = fc_feats.data.new(batch_size).long().zero_()
                rel_res = None
            elif sample_max:  # 其他情况下，贪婪编码，去最大概率即可
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:  # 依照概率取样
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).to("cpu")  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).to(torch.device("cpu"))
                it = torch.multinomial(prob_prev, 1).to(self.opts.device)
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing
            # 作为下一个时间点的输入
            it.requires_grad = False
            xt = self.word_embed(it) # 取前一个的采样结果作为输入

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)

                if unfinished.sum() == 0:
                    break

                it = it * unfinished.type_as(it)
                seq.append(it)  # seq[t] the input of t+2 time step 最后 seq是各个时间点（t-2个，少了BOS和EOS）的 batch * 1
                seqLogprobs.append(sampleLogprobs.view(-1)) # 同样，t-2个 batch * 1
                rel_ress.append(rel_res)

            output, state, rel_res = self.core(it, xt, fc_feats, att_feats, p_att_feats, state, stage_id)
            LMvocab = self.logit(output)
            if stage_id == 1 or stage_id == 2:
                # 只使用语言模型输出
                output = LMvocab
            else:
                # 使用线性融合
                rel_zero = LMvocab[:, 0].view(-1, 1)
                rel_others = LMvocab[:, self.nouns_size + 1:]  # barch * (vocab - nouns)
                rel = torch.cat((rel_zero, rel_res, rel_others), 1)

                output = self.fuse_coefficient * LMvocab + (1 - self.fuse_coefficient) * rel

            logprobs = F.log_softmax(output, 1)

        # return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
        if stage_id == 1 or stage_id == 2:
            return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1), None
        else:
            return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1), torch.cat([_.unsqueeze(1) for _ in rel_ress], 1)

    def beam_search(self, state, logprobs, *args, **kwargs):
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opt

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            # INPUTS:
            # logprobsf: probabilities augmented after diversity
            # beam_size: obvious
            # t        : time instant
            # beam_seq : tensor contanining the beams
            # beam_seq_logprobs: tensor contanining the beam logprobs
            # beam_logprobs_sum: tensor contanining joint logprobs
            # OUPUTS:
            # beam_seq : tensor containing the word indices of the decoded captions
            # beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            # beam_logprobs_sum : joint log-probability of each beam

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):  # for each column (word, essentially)
                for q in range(rows):  # for each beam expansion
                    # compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob.cpu()
                    candidates.append(dict(c=ix[q, c], q=q,
                                           p=candidate_logprob,
                                           r=local_logprob))
            candidates = sorted(candidates, key=lambda x: -x['p'])

            new_state = [_.clone() for _ in state]
            # beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
                # we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                # fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                # rearrange recurrent states
                for state_ix in range(len(new_state)):
                    #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']]  # dimension one is time step
                # append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c']  # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r']  # the raw logprob here
                beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        # start beam search
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)

        beam_seq = torch.LongTensor(self.max_caption_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.max_caption_length, beam_size).zero_()
        # running sum of logprobs for each beam
        beam_logprobs_sum = torch.zeros(beam_size)
        done_beams = []

        for t in range(self.max_caption_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            logprobsf = logprobs.data.float()  # lets go to CPU for more efficiency in indexing operations
            # suppress UNK tokens in the decoding
            logprobsf[:, logprobsf.size(1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000

            beam_seq, \
            beam_seq_logprobs, \
            beam_logprobs_sum, \
            state, \
            candidates_divm = beam_step(logprobsf,
                                        beam_size,
                                        t,
                                        beam_seq,
                                        beam_seq_logprobs,
                                        beam_logprobs_sum,
                                        state)

            for vix in range(beam_size):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 0 or t == self.max_caption_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(),
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix]
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # encode as vectors
            it = beam_seq[t]
            logprobs, state = self.get_logprobs_state(it.cuda(), *(args + (state,)))

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams


class BaseCore(nn.Module):
    def __init__(self, opts):
        super(BaseCore, self).__init__()
        self.opts = opts
        self.input_encoding_size = opts.input_encoding_size
        # self.rnn_type = opts.rnn_type
        self.rnn_size = opts.rnn_size
        self.num_layers = opts.num_layers
        self.drop_prob_lm = opts.dropout_prob
        self.fc_feat_size = opts.fc_feat_size
        self.att_feat_size = opts.att_feat_size
        self.att_hid_size = opts.att_hid_size

        self.vocabulary = opts.vocabulary  # 词汇索引到单词
        self.nouns = opts.nouns  # 名词单词到索引

        # Build a LSTM
        self.a2c = nn.Linear(self.rnn_size, 2 * self.rnn_size)  # 注意力上下文到c，因为要用maxout所以是两倍的rnn，512 - 1024
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)  # 单词嵌入结果到hid的嵌入，因为有三个门和两个maxout的c，512 - 2560
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)  # hid到hid，512 - 2560
        self.dropout = nn.Dropout(self.drop_prob_lm)  # dropout层

        self.attention = BaseAttention(opts)
        self.memory = BaseMemory(opts)
        self.relation = BaseRelation(opts)

    def forward(self, wordt, xt, fc_feats, att_feats, internal_att_feats, state, stage_id):
        att_res = self.attention(state[0][-1], att_feats, internal_att_feats)  # 先把att算出来 batch*rnn_size 64*512

        all_input_sums = self.i2h(xt) + self.h2h(
            state[0][-1])  # 先把x和h的线性嵌入加起来，准备计算三个门，前三个rnn_size batch*(5*rnn_size) 64*2560
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)  # 前三个部分 batch*(3*rnn_size) 64 * 1536
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)  # sigmoid变换 batch * (3*rnn_size) 64*1536
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)  # 输入门
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)  # 忘记门
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)  # 输出门

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + \
                       self.a2c(att_res)  # 后两个部分 batch*（2*rnn_size） 64*1024
        in_transform = torch.max(
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))  # maxout

        next_c = forget_gate * state[1][-1] + in_gate * in_transform  # c batch*rnn_size 64*512
        next_h = out_gate * torch.tanh(next_c)  # h batch * rnn_size 64 * 512
        output = self.dropout(next_h)  # 对输出，做一个dropout

        if stage_id == 1:
            # 第一阶段，直接使用语言模型的输出
            rel_res = None
        elif stage_id == 2:
            # 第二阶段，还是使用语言模型的输出，但是要存memory
            self.memory(att_res, wordt, stage_id)
            rel_res = None
        else:
            # 第三阶段，使用两个网络的模型输出，这里使用一个linear fuse（限名词）
            rel_res = self.relation(att_res, stage_id)  # 关系网络结果，batch * nouns_size 64 * 7668

        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state, rel_res

    def memory_ready(self):
        memory = torch.from_numpy(self.memory.get_memory()).to(self.opts.device, torch.float)
        self.relation.set_memory(memory)

    def memory_finish(self):
        # 完成算数平均值计算，并返回nunpy格式
        self.memory.finish()
        return self.memory.get_memory()

    def set_memory(self, memory):
        self.memory.set_memory(memory)


class BaseAttention(nn.Module):
    def __init__(self, opts):
        super(BaseAttention, self).__init__()
        self.rnn_size = opts.rnn_size
        self.att_hid_size = opts.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)  # hid到att向量的嵌入，512 - 512
        self.alpha_net = nn.Linear(self.att_hid_size, 1)  # 上下文最终表示, 512 - 1

    def forward(self, h, att_feats, internal_att_feats):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size  # 计算总att_size，(att*att), (14*14)
        att = internal_att_feats.view(-1, att_size,
                                      self.att_hid_size)  # 把上下文向量中间空间维度展开，batch*att_size*att_hid_size, 64 * 196 * 512

        # 处理hid和att
        att_h = self.h2att(h)  # 先把hid部分转换一下，batch * att_hid_size，64 * 512
        att_h = att_h.unsqueeze(1).expand_as(att)  # 针对每一个空间特征，复制一遍，batch * att_size * att_hid_size，64 * 196 * 512
        dot = att + att_h  # att部分和hid部分加在一起，batch * att_size * att_hid_size，64 * 196 * 512
        dot = torch.tanh(dot)  # tanh变化，batch * att_size * att_hid_size，64*196*512
        dot = dot.view(-1, self.att_hid_size)  # 变成两维度好计算，(batch * att_size) * att_hid_size，（64*196）*512
        dot = self.alpha_net(dot)  # 映射到区域注意力分数，(batch * att_size) * 1，（64*196）*1
        dot = dot.view(-1, att_size)  # 针对各个空间区域的分数，batch * att_size，64*196

        weight = F.softmax(dot, 1)  # 针对没一个区域的softmax，batch * att_size，64*196
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)  # 把空间区域部分展开 batch * att_size * rnn_size 64*196*512
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # 注意力结果 batch * rnn_size 64*512

        return att_res


class BaseRelation(nn.Module):
    def __init__(self, opts):
        super(BaseRelation, self).__init__()
        self.opts = opts
        self.vocab_size = opts.vocabulary_size  # 9486 词汇表
        self.nouns_size = opts.nouns_size  # 7668 名词表
        self.rnn_size = opts.rnn_size  # 512 rnn内部表示尺寸
        self.relation_pre_fc_size = opts.relation_pre_fc_size  # 特征拼接前的线性嵌入块的尺寸 512
        self.relation_post_fc_size = opts.relation_post_fc_size  # 特征拼接后的线性嵌入块的尺寸 512
        self.pre_fc = nn.Sequential(
            nn.Linear(self.rnn_size, self.relation_pre_fc_size),  # 前嵌入 512 - 512
            nn.ReLU()
        )

        self.post_fc = nn.Sequential(
            nn.Linear(2 * self.relation_pre_fc_size, self.relation_post_fc_size),  # 后嵌入 1024 - 512
            nn.ReLU()
        )

        self.rel_fc = nn.Sequential(
            nn.Linear(self.relation_post_fc_size, 1),  # 关系分数计算 512 - 1
            nn.Sigmoid()
        )

    def forward(self, att_res, stage_id):
        batch_size = att_res.size(0)  # batch
        memory_pre = self.pre_fc(self.nouns_memory)  # 处理所有类别的记忆表示 nouns_size * pre_fc_size 7668 * 128
        att_pre = self.pre_fc(att_res)  # 处理输入的注意力表示 batch * pre_fc_size 64 * 128

        memory_pre_ext = memory_pre.unsqueeze(0).repeat(batch_size, 1,
                                                        1)  # 在最外一维复制batch倍 batch*nouns_size*pre_fc 64*7668*128
        att_pre_ext = att_pre.unsqueeze(0).repeat(self.nouns_size, 1, 1)  # 同理 nouns_size * batch * pre_fc 7668*64*128
        att_pre_ext = torch.transpose(att_pre_ext, 0, 1)  # 转置一下注意力矩阵 batch*nouns_size*pre_fc 64*7668*128

        relation_pairs = torch.cat((memory_pre_ext, att_pre_ext), 2).view(-1, 2 * self.relation_pre_fc_size)  # 拼接到一块
        # (batch*nouns_size) * (2*pre_fc) (64*7668) * (2 * 128)

        relations_post = self.post_fc(relation_pairs) # 做一次非线性嵌入， (batch * nouns_size) * post_fc_size (64 * 7668) * 64
        relations = self.rel_fc(relations_post).view(-1, self.nouns_size) # 计算关系分数 (batch * nouns_size) * 1 -> batch * nouns_size (64 * 7688) * 1 -> 64 * 7668
        return relations

    def set_memory(self, memory):
        self.nouns_memory = memory[1:, :]
        self.nouns_memory.to(self.opts.device)


class BaseMemory(nn.Module):
    def __init__(self, opts):
        super(BaseMemory, self).__init__()
        self.vocabulary = opts.vocabulary  # 词汇索引到单词
        self.nouns = opts.nouns  # 名词单词到索引
        self.vocab_size = opts.vocabulary_size  # 9486 词汇表
        self.nouns_size = opts.nouns_size  # 7668 名词表
        self.memory_size = opts.memory_size  # 512 记忆表示长度
        self.nouns_memory = np.zeros((self.nouns_size + 1, self.memory_size), dtype=float)  # 7668 * 512 对所有名词类别的表示，初始化为0, 0位无效
        self.nouns_counter = np.zeros((self.nouns_size + 1, 1), dtype=int)  # 7668 * 1，存储记忆过程中的计数器，用于计算平均值

    def forward(self, att_res, wordt, stage_id):
        batch_size = att_res.size(0)
        # 写入模式，查询是否存在这个名词
        is_nouns = wordt.le(self.nouns_size).type_as(wordt) # batch * 1
        xt_nouns = (wordt * is_nouns)

        nouns = xt_nouns.to(torch.device("cpu")).numpy()
        att_ress = att_res.to(torch.device("cpu")).numpy()

        for i in range(batch_size):
            self.nouns_memory[nouns[i]] = self.nouns_memory[nouns[i]] + att_ress[i]
            self.nouns_counter[nouns[i]] += 1

    def get_memory(self):
        # 返回所有类别
        return self.nouns_memory

    def finish(self):
        # 整理所有的记忆，计算算数均值
        self.nouns_memory = self.nouns_memory / self.nouns_counter

    def set_memory(self, memory):
        self.nouns_memory = self.nouns_memory + memory
