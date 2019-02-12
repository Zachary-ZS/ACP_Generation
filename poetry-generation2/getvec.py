from parameters import *
TRAIN_DATA = r'data/train.txt'
VALFILE = r'data/val.txt'
VECFILE = 'sgns.sikuquanshu.bigram'
def getData():
    with open(TRAIN_DATA, 'r', encoding='utf-8') as ff:
        lines = ff.readlines()
    traindata = []
    for line in lines:
        for word in line.split():
            if word in traindata:
                None
            else:
                traindata.append(word)
    with open(VALFILE, 'r', encoding='utf-8') as ff:
        lines = ff.readlines()
    for line in lines:
        for word in line.split():
            if word in traindata:
                None
            else:
                traindata.append(word)
    return traindata


def getvec():
    with open(VECFILE, 'r', encoding='utf-8') as ff:
        lines = ff.readlines()
    tmp=[]
    for i in range(300):
        tmp.append(float(0))
    vec = []
    vec.append(tmp)
    vec.append(tmp)
    vec.append(tmp)
    index={}
    index['SOS']=0
    index[',']=1
    index['.']=2
    index2word=['SOS',',','.']
    flag=3
    data = getData()
    for line in lines:
        listl = line.strip('\n').split(' ')
        word = listl[0]
        if word in data and word != ',':
            index[word]=flag
            index2word.append(word)
            flag=flag+1
            vec.append([float(x) for x in listl[1:301]])
            data.remove(word)
        else:
            None
    data.remove('.')
    data.remove(',')
    for word in data:
        index[word]=flag;
        index2word.append(word)
        flag=flag+1
        vec.append(tmp)
    return index,vec,index2word

if __name__ == '__main__':
    getvec()