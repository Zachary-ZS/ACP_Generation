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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


START_TAG = "<START>"
STOP_TAG = "<STOP>"
DROPOUT = 0.3
LEARNING_RATE = 0.01
teacher_forcing_ratio = 0.5

MAX_LENGTH = 21

hidden_size = 300

USE_CLIP = False
CLIP = 10.0

USE_CUDA = True