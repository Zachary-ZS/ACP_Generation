from parameters import *
from getvec import *
import sys
import os
import numpy
from pypinyin import pinyin, Style, lazy_pinyin

TESTFILE = r'data/test.txt'

index, vec, index2word = getvec()


def indexesFromSentence(sentence):
    # return [lang.word2index[word] for word in sentence]
    return [index[word] for word in sentence]


def tensorFromSentence(sentence):
    indexes = indexesFromSentence(sentence)
    # indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def genInput():
    inputdata = []
    with open(TESTFILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for sen1 in lines:
        flg = 0
        tmplist = []
        for word in sen1.split():
            if flg == 7:
                break
            tmplist.append(word)
            flg = flg + 1
        inputdata.append(tmplist)
    return inputdata


def evaluate(encoder, decoder, sentence, sentencenum, max_length=21, rythm=False):
    with torch.no_grad():
        input_tensor = tensorFromSentence(sentence)
        input_tensor = input_tensor.cuda()
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        target_length = 7

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[0]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(target_length, max_length)

        if sentencenum == 2:
            CANDIDATE = 5
        elif sentencenum == 3:
            CANDIDATE = 3
        elif sentencenum == 4:
            CANDIDATE = 16
        RYTHMROOT = -1 if sentencenum in {2, 3} else 13
        # Rythm is now the YunMu to be rthymed, like 'en' for '门'
        RYTHM = lazy_pinyin(sentence[RYTHMROOT], style=Style.FINALS, strict=False)[0]

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            if rythm and di == target_length - 1:
                topval, topindex = decoder_output.squeeze(0).data.topk(CANDIDATE)
                bestindex = -1
                bestrythm = -1
                for i in range(CANDIDATE):
                    idx = topindex[i].item()
                    if idx == 0:
                        continue
                    if bestindex == -1:
                        bestindex = idx
                    if lazy_pinyin(index2word[idx], style=Style.FINALS, strict=False)[0] == RYTHM:
                        # Found the best rythm choice!
                        bestrythm = idx
                        break
                idx = bestrythm if bestrythm != -1 else bestindex
                decoded_words.append(index2word[idx])
            else:
                topv, topi = decoder_output.data.topk(1)
                decoded_words.append(index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words


if __name__ == '__main__':
    roundnum = input("round?")
    suffix = input("suffix?")
    encoder = torch.load('model/encoder_r' + roundnum + 's' + suffix + '.pkl')
    decoder = torch.load('model/decoder_r' + roundnum + 's' + suffix + '.pkl')
    wf = open('result' + roundnum+ '_' + suffix + 'northym.txt', 'w')
    wf2 = open('result'+ roundnum+ '_' + suffix + 'rthym.txt', 'w')
    with torch.no_grad():
        testdata = genInput()
        for sentence in testdata:
            tmpsentence = sentence
            for i in range(3):
                tmpwords = evaluate(encoder, decoder, tmpsentence, i + 2, rythm=False)
                tmpsentence = tmpsentence + tmpwords
            print(tmpsentence)
            wf.write(' '.join(tmpsentence) + '\n')
            tmpsentence = sentence
            for i in range(3):
                tmpwords = evaluate(encoder, decoder, tmpsentence, i + 2, rythm=True)
                tmpsentence = tmpsentence + tmpwords
            print(tmpsentence)
            wf2.write(' '.join(tmpsentence) + '\n')
