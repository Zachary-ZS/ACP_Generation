from parameters import *
import pickle

TRAINFILE = 'data/train.txt'
VALFILE = 'data/val.txt'
TESTFILE = 'data/test.txt'
TRAINLOAD = 'trainload.pkl'
VALLOAD = 'valload.pkl'
TESTLOAD = 'testload.pkl'
word_ix = {}
ix_word = []


def trainprepare():
    with open(TRAINFILE, 'r') as rf:
        lines = rf.readlines()
    with open(TRAINLOAD, 'wb') as wf:
        traindata = []
        for line in lines:
            line = line.strip().replace(" ", "")
            line = line.replace(',',' ').replace('.',' ')
            linelist = line.split()
            linelist = [list(line) for line in linelist]
            traindata.append(linelist)
        pickle.dump(traindata, wf)
    

def valprepare():
    with open(VALFILE, 'r') as rf:
        lines = rf.readlines()
    with open(VALLOAD, 'wb') as wf:
        valdata = []
        for line in lines:
            line = line.strip().replace(" ", "")
            line = line.replace(',',' ').replace('.',' ')
            linelist = line.split()
            linelist = [list(line) for line in linelist]
            valdata.append(linelist)
        pickle.dump(valdata, wf)
    

def testprepare():
    with open(TESTFILE, 'r') as rf:
        lines = rf.readlines()
    with open(TESTLOAD, 'wb') as wf:
        testdata = []
        for line in lines:
            line = line.strip().replace(" ", "")
            line = line.replace(',','')
            linelist = list(line)
            testdata.append(linelist)
        pickle.dump(testdata, wf)


def buildindex():
    poems = []
    with open(TRAINLOAD, 'rb') as readf:
        poems = pickle.load(readf)

    word_ix = {'sos': 0, '，': 1, '。': 2,}
    ix_word = ['sos', '，', '。',]
    for poem in poems:
        for sentence in poem:
            for char in sentence:
                if char not in word_ix:
                    ix_word.append(char)
                    word_ix[char] = len(word_ix)

    with open('word_ix.pkl', 'wb') as wf:
        pickle.dump(word_ix, wf)
    with open('ix_word.pkl', 'wb') as wf:
        pickle.dump(ix_word, wf)

def trainsplit():
    with open(TRAINLOAD, 'rb') as rf:
        traindata = pickle.load(rf)
    pairs1 = []
    pairs2 = []
    pairs3 = []
    
    for poem in traindata:
        # poem is in the format of list[4]
        pairs1.append([poem[0] + ['，'], poem[1]])
        pairs2.append([poem[0] + ['，'] + poem[1] + ['。'], poem[2]])
        pairs3.append([poem[0] + ['，'] + poem[1] + ['。'] + poem[2] + ['，'], poem[3]])
    with open('pairs1.pkl', 'wb') as f1:
        pickle.dump(pairs1, f1)
    with open('pairs2.pkl', 'wb') as f2:
        pickle.dump(pairs2, f2)
    with open('pairs3.pkl', 'wb') as f3:
        pickle.dump(pairs3, f3)

    with open('pairs3.pkl', 'rb') as f1:
        wepair1 = pickle.load(f1)
        print(wepair1[0])
        print(wepair1[3])
        
def valsplit():
    with open(VALLOAD, 'rb') as rf:
        valdata = pickle.load(rf)
    pairs1 = []
    pairs2 = []
    pairs3 = []
    
    for poem in valdata:
        # poem is in the format of list[4]
        pairs1.append([poem[0] + ['，'], poem[1]])
        pairs2.append([poem[0] + ['，'] + poem[1] + ['。'], poem[2]])
        pairs3.append([poem[0] + ['，'] + poem[1] + ['。'] + poem[2] + ['，'], poem[3]])
    with open('valpairs1.pkl', 'wb') as f1:
        pickle.dump(pairs1, f1)
    with open('valpairs2.pkl', 'wb') as f2:
        pickle.dump(pairs2, f2)
    with open('valpairs3.pkl', 'wb') as f3:
        pickle.dump(pairs3, f3)

    with open('valpairs3.pkl', 'rb') as f1:
        wepair1 = pickle.load(f1)
        print(wepair1[0])
        print(wepair1[3])

if __name__ == '__main__':
    #buildindex()
    valsplit()
