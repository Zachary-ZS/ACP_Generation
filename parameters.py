
import string
import re
import sys
import pickle
import random
import time
import datetime
import math
import numpy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

TRAININPUT = 'trainload.pkl'
VALINPUT = 'valload.pkl'
#    train corpus input, with the format of list[poetrynum],
# each item in list representing a poem is a 4 * 7 matrix
# for one character at each position, like this:
#   list[0]:
# [['不', '数', '僧', '繇', '只', '数', '吴'],
#  ['丹', '青', '纪', '出', '活', '形', '模'],
#  ['问', '君', '写', '尽', '人', '多', '少'],
#  ['写', '得', '人', '心', '一', '片', '无']]
# 
# VALINPUT has the same format


SOS_token = 0

EMBEDDING_DIM = 300
HIDDEN_DIM = 20
DROPOUT = 0.3
LEARNING_RATE = 0.01
LAYERS = 1
batch_size = 40
EPOCHNUM = 50000
REPORTNUM = 200
SAVENUM = 600
attn_model = 'dot'
GRUorLSTM = 'lstm'

USE_CLIP = False
CLIP = 10.0
PRETRAIN = False
PRETRAINEDFILE = 'wordvec.pkl'
# Wether to use the pretrained word-vectors

USE_CUDA = True


# Masked cross entropy loss, copied from former codes.
def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    # seq_range = torch.range(0, max_len-1).long()
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def masked_cross_entropy(logits, target, length, USE_CUDA):
    length = Variable(torch.LongTensor(length))
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat,dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss