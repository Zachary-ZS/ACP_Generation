import sys
import os
import re
import unicodedata
import random

TRAIN_DATA = r'data/train.txt'
SOS_token = 0

def getData():
    with open(TRAIN_DATA, 'r', encoding='utf-8') as ff:
        lines = ff.readlines()
    traindata = []
    onepoem = []
    for line in lines:
        cnt = 0
        tmplist = []
        for word in line.split():
            if cnt == 7:
                cnt = 0
                onepoem.append(tmplist)
                tmplist = []
            else:
                tmplist.append(word)
                cnt = cnt + 1
        traindata.append(onepoem)
        onepoem = []
    for item in traindata:
        print(item)
    return traindata

def readpoems():
    print("Reading lines...")
    with open(TRAIN_DATA, 'r', encoding='utf-8') as ff:
        lines = ff.readlines()
    traindata = []
    onepoem = []
    for line in lines:
        cnt = 0
        tmplist = []
        for word in line.split():
            if cnt == 7:
                cnt = 0
                if len(onepoem) == 0:
                    None
                else:
                    traindata.append([list(onepoem), tmplist])
                onepoem = onepoem + tmplist
                tmplist = []
            else:
                tmplist.append(word)
                cnt = cnt + 1
        onepoem = []

    return traindata





