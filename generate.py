from parameters import *
from train import generate_a_poem, generate_a_sen
import getopt

opts, args = getopt.getopt(sys.argv[1:], 'ptn', ["part", "total", "name="])
PART = False
TOTAL = False
MODELNAME = ''
for op, val in opts:
    if op in ("--part", "-p"):
        PART = True
    elif op in ("--total", "-t"):
        TOTAL = True
    elif op in ("--name", "-n"):
        MODELNAME = val
modelnum = int(
        input("Please input the number for the model %s :" % (MODELNAME)))
GRUorLSTM = input("It's gru or lstm?")
# if GRUorLSTM == 'gru':
#     from model1 import *
# else:
#     from model import *
from model import *


if __name__ == '__main__':

    

    print("Loading model...")
    encoder = torch.load('model/%s_encoder%d.pkl' % (MODELNAME, modelnum))
    decoder = torch.load('model/%s_decoder%d.pkl' % (MODELNAME, modelnum))

    pre = input("Using pretrained vectors? __pre__ or __not__: ")

    print("Loading indexes...")
    if pre == 'pre':
        with open('word-vec/index.pkl', 'rb') as wif:
            word_ix = pickle.load(wif)
        ix_word = dict(zip(word_ix.values(),word_ix.keys()))
    else:
        with open('word-vec/word_ix.pkl', 'rb') as wif:
            word_ix = pickle.load(wif)
        with open('word-vec/ix_word.pkl', 'rb') as iwf:
            ix_word = pickle.load(iwf)
    print("Loading data...")
    with open('testload.pkl', 'rb') as dataf:
        sentences = pickle.load(dataf)

    if TOTAL:
        # generate all poems
        OUTFILE1 = "result/%s_%d_rythm.txt" % (MODELNAME, modelnum)
        OUTFILE2 = "result/%s_%d_nonrythm.txt" % (MODELNAME, modelnum)
        with open(OUTFILE1, 'w') as wf:
            for sentence in sentences:
                with torch.no_grad():
                    poem = generate_a_poem(
                        sentence, encoder, decoder, word_ix, ix_word, rythm=True)
                    wf.write(''.join(poem) + '\n')
        with open(OUTFILE2, 'w') as wf:
            for sentence in sentences:
                with torch.no_grad():
                    poem = generate_a_poem(
                        sentence, encoder, decoder, word_ix, ix_word, rythm=False)
                    wf.write(''.join(poem) + '\n')

    if PART:
        # randomly generate 4 poems
        for i in range(4):
            with torch.no_grad():
                sentence = random.choice(sentences)
                poem = generate_a_poem(
                    sentence, encoder, decoder, word_ix, ix_word)
                print(''.join(poem))
