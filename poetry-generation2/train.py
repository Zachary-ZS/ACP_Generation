from parameters import *
from data import *
from model import *
from getvec import *

pairs = readpoems()
index, vec, index2word = getvec()
torch.cuda.set_device(0)


def indexesFromSentence(sentence):
    # return [lang.word2index[word] for word in sentence]
    return [index[word] for word in sentence]


def tensorFromSentence(sentence):
    indexes = indexesFromSentence(sentence)
    # indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(pair[0])
    target_tensor = tensorFromSentence(pair[1])
    return (input_tensor, target_tensor)


# for item in pairs:
#     print(item)
# print(tensorsFromPair(pairs[-1])[0])
# print(tensorsFromPair(pairs[-1])[0].size())
# print(tensorsFromPair(pairs[-2]))
encoder1 = EncoderRNN(len(vec), hidden_size, vec).to(device)
attn_decoder1 = AttnDecoderRNN(
    hidden_size, len(vec), vec, dropout_p=0.1).to(device)

# Gen sentence
def gen_sentence(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        target_length = 7
        # Set to not-training mode to disable dropout
        encoder.train(False)
        decoder.train(False)

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(target_length, max_length)

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            # if topi.item() == EOS_token:
            #     decoded_words.append('<EOS>')
            #     break
            # else:
            decoded_words.append(index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        # Set back to training mode
        encoder.train(True)
        decoder.train(True)

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=6):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0], file=f)
        print('=', pair[1], file=f)
        output_words, attentions = gen_sentence(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence, file=f)
        print('', file=f)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            # if decoder_input.item() == EOS_token:
            #     break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01, save_every=20000):
    '''
    :param n_iters: randomly choose n_iter pairs for training
    :param print_every: print time and loss when iter%print_every ==0
    :param save_every: save the model when iter%save_every ==0
    '''
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()
    lentrain = len(training_pairs)

    for round in range(200):
        for iter in range(1, lentrain):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print("Round %d and sentence pair %d" % (round, iter))

                print('%s (%d %d%%) %.4f' % (
                    timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

                print("Round %d and sentence pair %d" % (round, iter), file=f)

                print('%s (%d %d%%) %.4f' % (
                    timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg), file=f)

                evaluateRandomly(encoder1, attn_decoder1)
            if iter % save_every == 0:
                torch.save(encoder, "model/encoder_r%ds%d.pkl" % (round, iter))
                torch.save(decoder, "model/decoder_r%ds%d.pkl" % (round, iter))


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# opts, argvs = getopt.getopt(sys.argv[1:], 'n:')
# MODELNAME = ''
# for op, val in opts:
#     if op == '-n':
#         MODELNAME = val
f = open("record-.txt", 'w')
trainIters(encoder1, attn_decoder1, 75000, print_every=1000)
