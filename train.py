from parameters import *
from model import *
import getopt
from pypinyin import pinyin, Style, lazy_pinyin


def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    # Run encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    
    # Decoder input preparation
    decoder_input = torch.LongTensor([SOS_token] * batch_size)
    if GRUorLSTM == 'lstm':
        decoder_hidden = (encoder_hidden[0][:decoder.n_layers], encoder_hidden[1][:decoder.n_layers])
    else:
        decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Sequence to store the output in
    target_length = target_lengths[0]
    all_decoder_outputs = torch.zeros(target_length, batch_size, decoder.output_size)
    
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
        
    # Run decoder one step at a time.
    for i in range(target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
        
        all_decoder_outputs[i] = decoder_output
        # using teacher forcing
        decoder_input = target_batches[i]
    
    loss = masked_cross_entropy(all_decoder_outputs.transpose(0, 1).contiguous(),         target_batches.transpose(0, 1).contiguous(),        target_lengths, USE_CUDA)
    # BackP
    loss.backward()

    if USE_CLIP:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), CLIP)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)
    
    # update
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item()

def evaluate_batch(input_batches, input_lengths, encoder, decoder, word_ix, ix_word, target_batches=None, target_lengths=None):
    target_length = target_lengths[0]
    # Run encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([SOS_token] * batch_size)  # SOS
    if GRUorLSTM == 'gru':
        decoder_hidden = encoder_hidden[:decoder.n_layers]
    else:
        decoder_hidden = (encoder_hidden[0][:decoder.n_layers], encoder_hidden[1][:decoder.n_layers])
    # decoder_hidden = decoder_hidden.contiguous()

    # Store output words and attention states
    
    # decoded_words = []
    #decoder_attentions = torch.zeros(target_length + 1, target_length + 1)
    all_decoder_outputs = torch.zeros(target_length, batch_size, decoder.output_size)

    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
    
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

        all_decoder_outputs[di] = decoder_output
        #decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data
        
        # generate new char
        topv, topi = decoder_output.data.topk(1)
        #ni = topi.item()
        #decoded_words.append(ix_word[ni])
        # Next input
        decoder_input = topi.squeeze().detach()

    loss = masked_cross_entropy(all_decoder_outputs.transpose(0, 1).contiguous(), target_batches.transpose(0, 1).contiguous(), target_lengths, USE_CUDA)
    
    return loss.item()#, decoded_words

def evaluate(pairs1, pairs2, pairs3, encoder, decoder, word_ix, ix_word):
    '''
    Args:\\
    `pairs1-3`: separately the traindata_pair for one-one, two-one, three-one generation.\\
    `encoder, decoder`\\
    `word_ix, ix_word`\\
    Used to evaluate the result on `val`
    '''
    input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size, pairs1, word_ix)
    loss1 = evaluate_batch(input_batches, input_lengths, encoder, decoder, word_ix, ix_word, target_batches, target_lengths)

    input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size, pairs2, word_ix)
    loss2 = evaluate_batch(input_batches, input_lengths, encoder, decoder, word_ix, ix_word, target_batches, target_lengths)

    input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size, pairs3, word_ix)
    loss3 = evaluate_batch(input_batches, input_lengths, encoder, decoder, word_ix, ix_word, target_batches, target_lengths)
    print("------ Result on val: ------", file=f)
    print("Val Losses: pair1: %lf    pair2: %lf    pair3: %lf" % (loss1, loss2, loss3), file=f)

def generate_a_sen(sentence, encoder, decoder, word_ix, ix_word, sentencenum, rythm=True, target_length=7):
    '''
    Args:\\
    sentence : input as a list of characters, like `['日','暮','苍','山','远']`, prepare_sequence can make it in the format of index list\\
    sentencenum : show which sentence it is in a poem\\
    rythm : wether to rythm or not\\
    return : A list in sizeof target_length, represents the sentence
    '''
    input_seq = [prepare_seq(sentence, word_ix)]
    input_lengths = [len(s) for s in input_seq]
    input_batches = torch.LongTensor(input_seq).transpose(0, 1)
    
    if USE_CUDA:
        input_batches = input_batches.cuda()
    # Run encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    # Prepare decoder inputs, be sure we now generate only one sen.
    decoder_input = torch.LongTensor([SOS_token])
    decoder_hidden = encoder_hidden
    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store new words
    decoded_words = []

    if sentencenum == 2:
        CANDIDATE = 5
    elif sentencenum == 3:
        CANDIDATE = 3
    elif sentencenum == 4:
        CANDIDATE = 16
    RYTHMROOT = -2 if sentencenum in {2, 3} else 14
    # Rythm is now the YunMu to be rthymed, like 'en' for '门'
    RYTHM = lazy_pinyin(sentence[RYTHMROOT], style=Style.FINALS, strict=False)[0]

    # Run decoder, one step at a time
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

        # now use the time_step_best way to generate
        if rythm and di == target_length - 1:
            topval, topindex = decoder_output.squeeze(0).data.topk(CANDIDATE)
            bestindex = -1
            bestrythm = -1
            for i in range(CANDIDATE):
                idx = topindex[i].item()
                if idx == SOS_token:
                    continue
                if bestindex == -1:
                    bestindex = idx
                if lazy_pinyin(ix_word[idx], style=Style.FINALS, strict=False)[0] == RYTHM:
                    # Found the best rythm choice!
                    bestrythm = idx
                    break
            idx = bestrythm if bestrythm != -1 else bestindex
        else:
            topval, topindex = decoder_output.squeeze(0).data.topk(2)
            if topindex[0].item() != SOS_token:
                idx = topindex[0].item()
            else:
                idx = topindex[1].item()        
        decoded_words.append(ix_word[idx])

        # Next input
        decoder_input = torch.LongTensor([idx])
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
    
    return decoded_words

def generate_a_poem(sentence, encoder, decoder, word_ix, ix_word, rythm=True):
    '''
    Args:\\
    sentence : input as a list of characters, like `['日','暮','苍','山','远']`, prepare_sequence can make it in the format of index list\\
    return : the whole poem, in the format of a list of size(32):
    [
     '不', '数', '僧', '繇', '只', '数', '吴', '，'
     '丹', '青', '纪', '出', '活', '形', '模', '。'
     '问', '君', '写', '尽', '人', '多', '少', '，'
     '写', '得', '人', '心', '一', '片', '无', '。']
    '''
    sentence = sentence + ['，']
    tmp = generate_a_sen(sentence, encoder, decoder, word_ix, ix_word, sentencenum = 2, rythm = rythm)
    sentence = sentence + tmp + ['。']
    tmp = generate_a_sen(sentence, encoder, decoder, word_ix, ix_word, sentencenum = 3, rythm = rythm)
    sentence = sentence + tmp + ['，']
    tmp = generate_a_sen(sentence, encoder, decoder, word_ix, ix_word, sentencenum = 4, rythm = rythm)
    sentence = sentence + tmp + ['。']
    return sentence


def random_batch(batch_size, pairs, word_ix):
    input_seq = []
    target_seq = []

    # generate random batch_size pairs.
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seq.append(prepare_seq(pair[0], word_ix))
        target_seq.append(prepare_seq(pair[1], word_ix))
    
    # zip into pairs
    tmppairs = sorted(zip(input_seq, target_seq), key=lambda p: len(p[0]), reverse=True)
    input_seq, target_seq = zip(*tmppairs)

    input_lengths = [len(sentence) for sentence in input_seq]
    target_lengths = [len(sentence) for sentence in target_seq]
    # Here we needn't pad the sentences, cuz we have restricted 
    # the input with a unitive length.

    # size: from (batchsize * maxlen) to (maxlen * batchsize)
    input_batches = torch.LongTensor(input_seq).transpose(0, 1)
    target_batches = torch.LongTensor(target_seq).transpose(0, 1)
    if USE_CUDA:
        input_batches = input_batches.cuda()
        target_batches = target_batches.cuda()

    return input_batches, input_lengths, target_batches, target_lengths


def prepare_seq(sentence, word_ix):
    # reurns the list of indexes.
    idxs = []
    for word in sentence:
        if word in word_ix:
            idxs.append(word_ix[word])
        else:
            idxs.append(len(word_ix) - 1)
        # Remaining to be modified.
        # Now for unk words just set it to the last word in word-dict.
    # return torch.tensor(idxs, dtype=torch.long)
    # idxs = idxs + [EOS_token]
    return idxs
    

if __name__ == '__main__':

    MODELNAME = ""
    # Arguments: -d:dropout   -e:embedding_dim  
    #            -c:clip_val  -l:l-rate     -n:model_name
    opts, args = getopt.getopt(sys.argv[1:], 'd:e:c:l:n:',["pre"])
    print("Now using the set of parameters: ")
    for op, val in opts:
        if op == '-d':
            DROPOUT = float(val)
            print("Dropout: %lf" % (DROPOUT), end='  ')
        elif op == '-e':
            EMBEDDING_DIM = int(val)
            print("EMBEDDING-dim: %d" % (EMBEDDING_DIM), end='  ')
        elif op == '-c':
            USE_CLIP = True
            CLIP = float(val)
            print("Clip: %lf" % (CLIP), end='  ')
        elif op == '-l':
            LEARNING_RATE = float(val)
            print("Learning Rate: %lf" % (LEARNING_RATE), end='  ')
        elif op == '-n':
            MODELNAME = val
            print("Model Name: %s" % (MODELNAME), end='  ')
        elif op == "--pre":
            PRETRAIN = True
    print("\nOther parameters are the default vals in --parameters.py--.")
    print("Loading data...")

    # with open(TRAININPUT, 'rb') as trainf:
    #     traindata = pickle.load(trainf)
    with open(VALINPUT, 'rb') as valf:
        valdata = pickle.load(valf)
    if PRETRAIN:
        with open('word-vec/vec.pkl', 'rb') as vecf:
            vectrix = pickle.load(vecf)
        with open('word-vec/index.pkl', 'rb') as idxf:
            word_ix = pickle.load(idxf)
        # idx is the dict of word-ix
        ix_word = dict(zip(word_ix.values(),word_ix.keys()))
        
    else:
        with open('word-vec/word_ix.pkl', 'rb') as wif:
            word_ix = pickle.load(wif)
        with open('word-vec/ix_word.pkl', 'rb') as iwf:
            ix_word = pickle.load(iwf)
    vocal_size = len(word_ix)
    with open('data/pairs1.pkl', 'rb') as f1:
        pairs1 = pickle.load(f1)
    with open('data/pairs2.pkl', 'rb') as f2:
        pairs2 = pickle.load(f2)
    with open('data/pairs3.pkl', 'rb') as f3:
        pairs3 = pickle.load(f3)
    with open('data/valpairs1.pkl', 'rb') as f1:
        valpairs1 = pickle.load(f1)
    with open('data/valpairs2.pkl', 'rb') as f2:
        valpairs2 = pickle.load(f2)
    with open('data/valpairs3.pkl', 'rb') as f3:
        valpairs3 = pickle.load(f3)
        
    print("Finished loading data.")

    f = open('record-%s.txt' % (MODELNAME), 'w')
    print("--------- MODEL:: %s ---------" % (MODELNAME), file=f)
    
    
    encoder = EncoderRNN(vocal_size, EMBEDDING_DIM, GRUorLSTM, LAYERS, dropout=DROPOUT, PRETRAIN=PRETRAIN, A=vectrix)
    decoder = DecoderRNN(attn_model, EMBEDDING_DIM, vocal_size, GRUorLSTM, LAYERS, dropout=DROPOUT, PRETRAIN=PRETRAIN, A=vectrix)
    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=LEARNING_RATE)

    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()
        
    epoch = 0
    total_loss1 = 0.
    total_loss2 = 0.
    total_loss3 = 0.
    while epoch < EPOCHNUM:
        epoch += 1
        print("Processing the %d epoch." % (epoch))
        
        # to be fair let's batch in every pairs of different lengths each time, that's to say, now an epoch means 3*batch_size pairs tobe trained.
        input_batches, input_length, target_batches, target_length = random_batch(batch_size, pairs1, word_ix)
        loss1 = train(input_batches, input_length, target_batches, target_length, encoder, decoder, encoder_optimizer, decoder_optimizer)

        input_batches, input_length, target_batches, target_length = random_batch(batch_size, pairs2, word_ix)
        loss2 = train(input_batches, input_length, target_batches, target_length, encoder, decoder, encoder_optimizer, decoder_optimizer)

        input_batches, input_length, target_batches, target_length = random_batch(batch_size, pairs3, word_ix)
        loss3 = train(input_batches, input_length, target_batches, target_length, encoder, decoder, encoder_optimizer, decoder_optimizer)
        
        total_loss1 += loss1
        total_loss2 += loss2
        total_loss3 += loss3

        print("\nNow we've finished %d epoches." % (epoch))
        print("Current Losses: pair1: %lf    pair2: %lf    pair3: %lf" % (loss1, loss2, loss3))
        print("Average Losses: pair1: %lf    pair2: %lf    pair3: %lf" % (total_loss1 / epoch, total_loss2 / epoch, total_loss3 / epoch))

        if epoch % REPORTNUM == 0:
            print("\nNow we've finished %d epoches." % (epoch), file=f)
            print("Current Losses: pair1: %lf    pair2: %lf    pair3: %lf" % (loss1, loss2, loss3), file=f)
            print("Average Losses: pair1: %lf    pair2: %lf    pair3: %lf" % (total_loss1 / epoch, total_loss2 / epoch, total_loss3 / epoch), file=f)

            print("\nNow we've finished %d epoches." % (epoch))
            print("Current Losses: pair1: %lf    pair2: %lf    pair3: %lf" % (loss1, loss2, loss3))
            print("Average Losses: pair1: %lf    pair2: %lf    pair3: %lf" % (total_loss1 / epoch, total_loss2 / epoch, total_loss3 / epoch))

            # Val test
            with torch.no_grad():
                # evaluate loss result
                evaluate(valpairs1, valpairs2, valpairs3, encoder, decoder, word_ix, ix_word)
                # What's more, let's generate some samples.
                print("Some generated smaples:", file=f)
                for i in range(4):
                    pair = random.choice(valpairs3)
                    inputseq = pair[0][:7]
                    poem = generate_a_poem(inputseq, encoder, decoder, word_ix, ix_word)
                    groundtruth = pair[0] + pair[1]
                    print(">>>in>>> %s", ''.join(groundtruth), file=f)
                    print(">>>out>> %s", ''.join(poem), file=f)
                    
                    


        if epoch % SAVENUM == 0:
            torch.save(encoder, 'model/%s_encoder%d.pkl' % (MODELNAME, epoch))
            torch.save(decoder, 'model/%s_decoder%d.pkl' % (MODELNAME, epoch))
            
                


    