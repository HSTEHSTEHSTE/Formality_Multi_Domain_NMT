#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way. 
"""


from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
from io import open

import matplotlib
#if you are running on the gradx/ugradx/ another cluster, 
#you will need the following line
#if you run on a local machine, you can comment it out
#matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid, 
# it can be very easy to confict with other people's jobs.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15

class JpVocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence:
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence||first tgt sentence||formality
    second src sentence||second tgt sentence||formality
    into a list of things like
    [("first src sentence", "first tgt sentence", "formality"),
     ("second src sentence", "second tgt sentence", "formality")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into triplets
    triplets = [l.split('||') for l in lines]
    return triplets


def make_vocabs(src_lang_code, tgt_lang_code, corpus_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = JpVocab(tgt_lang_code)

    triplets = split_lines(corpus_file)
    train_triplets = triplets[:8*len(triplets)//10]

    for triplet in train_triplets:
        src_vocab.add_sentence(triplet[1])
        tgt_vocab.add_sentence(triplet[0])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################

def tensor_from_jp_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence:
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_triplet(src_vocab, tgt_vocab, triplets):
    """creates a tensor from a raw sentence triplet
    """
    input_tensors = []
    target_tensors = []
    for triplet in triplets:
      current_input_tensor = tensor_from_jp_sentence(src_vocab, triplet[1])
      current_target_tensor = tensor_from_sentence(tgt_vocab, triplet[0])
      input_tensors.append(F.pad(current_input_tensor, (0, 0, 0, MAX_LENGTH - current_input_tensor.size(0)), value = 1))
      target_tensors.append(F.pad(current_target_tensor, (0, 0, 0, MAX_LENGTH - current_target_tensor.size(0)), value = 1))
    input_tensor = torch.stack(input_tensors, dim = 1)
    target_tensor = torch.stack(target_tensors, dim = 1)
    return input_tensor, target_tensor

######################################################################

class LSTM(nn.Module):
    """the class for general-purpose LSTM
    """

    # todo: no batch support

    def __init__(self, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_directions = 1

        self.weight_input_hidden_i = nn.Parameter(torch.zeros([self.num_layers, self.num_directions * self.hidden_size, self.hidden_size])).cuda()
        self.weight_hidden_hidden_i = nn.Parameter(torch.zeros([self.num_layers, self.hidden_size, self.hidden_size])).cuda()
        self.bias_input_hidden_i = nn.Parameter(torch.zeros([self.num_layers, 1, self.hidden_size])).cuda()
        self.bias_hidden_hidden_i = nn.Parameter(torch.zeros([self.num_layers, 1, self.hidden_size])).cuda()

        self.weight_input_hidden_f = nn.Parameter(torch.zeros([self.num_layers, self.hidden_size, self.num_directions * self.hidden_size])).cuda()
        self.weight_hidden_hidden_f = nn.Parameter(torch.zeros([self.num_layers, self.hidden_size, self.hidden_size])).cuda()
        self.bias_input_hidden_f = nn.Parameter(torch.zeros([self.num_layers, 1, self.hidden_size])).cuda()
        self.bias_hidden_hidden_f = nn.Parameter(torch.zeros([self.num_layers, 1, self.hidden_size])).cuda()

        self.weight_input_hidden_g = nn.Parameter(torch.zeros([self.num_layers, self.hidden_size, self.num_directions * self.hidden_size])).cuda()
        self.weight_hidden_hidden_g = nn.Parameter(torch.zeros([self.num_layers, self.hidden_size, self.hidden_size])).cuda()
        self.bias_input_hidden_g = nn.Parameter(torch.zeros([self.num_layers, 1, self.hidden_size])).cuda()
        self.bias_hidden_hidden_g = nn.Parameter(torch.zeros([self.num_layers, 1, self.hidden_size])).cuda()

        self.weight_input_hidden_o = nn.Parameter(torch.zeros([self.num_layers, self.hidden_size, self.num_directions * self.hidden_size])).cuda()
        self.weight_hidden_hidden_o = nn.Parameter(torch.zeros([self.num_layers, self.hidden_size, self.hidden_size])).cuda()
        self.bias_input_hidden_o = nn.Parameter(torch.zeros([self.num_layers, 1, self.hidden_size])).cuda()
        self.bias_hidden_hidden_o = nn.Parameter(torch.zeros([self.num_layers, 1, self.hidden_size])).cuda()

        nn.init.normal_(self.weight_input_hidden_i)
        nn.init.normal_(self.weight_hidden_hidden_i)
        nn.init.normal_(self.bias_input_hidden_i)
        nn.init.normal_(self.bias_hidden_hidden_i)

        nn.init.normal_(self.weight_input_hidden_f)
        nn.init.normal_(self.weight_hidden_hidden_f)
        nn.init.normal_(self.bias_input_hidden_f)
        nn.init.normal_(self.bias_hidden_hidden_f)
        
        nn.init.normal_(self.weight_input_hidden_g)
        nn.init.normal_(self.weight_hidden_hidden_g)
        nn.init.normal_(self.bias_input_hidden_g)
        nn.init.normal_(self.bias_hidden_hidden_g)

        nn.init.normal_(self.weight_input_hidden_o)
        nn.init.normal_(self.weight_hidden_hidden_o)
        nn.init.normal_(self.bias_input_hidden_o)
        nn.init.normal_(self.bias_hidden_hidden_o)

        # We use nn.Dropout to implement dropout
        self.dropout_layer = nn.Dropout(p = self.dropout)

        self.initial_cell_state = torch.zeros(1, 1, self.hidden_size, device=device).cuda()
        # nn.init.normal_(self.initial_cell_state)
        self.cell_state = self.initial_cell_state.clone()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def get_initial_cell_state(self):
        return self.initial_cell_state

    def forward(self, input_in, hidden, cell_state = None):
        if cell_state is None:
            cell_state = self.cell_state
        input_tensor = input_in
        for current_layer in range(0, self.num_layers):
            i_t = self.sigmoid(torch.matmul(input_tensor, self.weight_input_hidden_i[current_layer]) + self.bias_input_hidden_i[current_layer] + torch.matmul(hidden, self.weight_hidden_hidden_i[current_layer]) + self.bias_hidden_hidden_i[current_layer])
            f_t = self.sigmoid(torch.matmul(input_tensor, self.weight_input_hidden_f[current_layer]) + self.bias_input_hidden_f[current_layer] + torch.matmul(hidden, self.weight_hidden_hidden_f[current_layer]) + self.bias_hidden_hidden_f[current_layer])
            g_t = self.tanh(torch.matmul(input_tensor, self.weight_input_hidden_g[current_layer]) + self.bias_input_hidden_g[current_layer] + torch.matmul(hidden, self.weight_hidden_hidden_g[current_layer]) + self.bias_hidden_hidden_g[current_layer])
            o_t = self.sigmoid(self.bias_input_hidden_o[current_layer] + torch.matmul(hidden, self.weight_hidden_hidden_o[current_layer]) + self.bias_hidden_hidden_o[current_layer] + torch.matmul(input_tensor, self.weight_input_hidden_o[current_layer]))

            cell_state = f_t * cell_state + i_t * g_t
            input_tensor = o_t * self.tanh(cell_state)

            input_tensor = self.dropout_layer(input_tensor)

        hidden = input_tensor
        return o_t, hidden, cell_state

######################################################################

class EncoderRNN(nn.Module):
    """the class for the enoder RNN
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size # vocab size
        self.hidden_size = hidden_size
        """Initilize a word embedding and bi-directional LSTM encoder
        For this assignment, you should *NOT* use nn.LSTM. 
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        You should make your LSTM modular and re-use it in the Decoder.
        """
        self.embedder = nn.Embedding(self.input_size, self.hidden_size)
        self.LSTM = nn.LSTM(self.hidden_size, self.hidden_size, 1, dropout = 0)

    def forward(self, input, hidden, cell_state):
        """runs the forward pass of the encoder
        input: the previous token
        returns the output and the hidden state
        """
        input_embedded = torch.transpose(self.embedder(input), 0, 1)
        output, (new_hidden, cell_state) = self.LSTM(input_embedded, (hidden, cell_state))
        return output, new_hidden, cell_state

    def get_initial_hidden_state(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device).cuda()

    def get_initial_cell_state(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device).cuda()

    def reset_cell_state(self, batch_size):
        return self.get_initial_cell_state(batch_size)

class AttnDecoderRNN(nn.Module):
    """the class for the decoder 
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.2, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.dropout = nn.Dropout(self.dropout_p)

        
        """Initilize your word embedding, decoder LSTM, and weights needed for your attention here
        """
        # i'm 80.1% sure that these input sizes are right
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn_0 = nn.Linear(self.hidden_size, self.max_length)
        self.sigmoid = nn.Sigmoid()

        self.embedder = nn.Embedding(self.output_size, self.hidden_size)
        self.LSTM = nn.LSTM(self.hidden_size, self.hidden_size, 1, dropout = self.dropout_p)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)

        self.input_processor = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.logsoftmax = nn.LogSoftmax(dim = 2)
        self.softmax = nn.Softmax(dim = 2)

    def forward(self, input, hidden, decoder_context, encoder_outputs, cell_state):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights
        
        Dropout (self.dropout) should be applied to the word embeddings.
        """

        concatenated_input = torch.cat([torch.transpose(self.embedder(input), 0, 1), decoder_context], 2)
        processed_input = self.input_processor(concatenated_input)

        output, (new_hidden, cell_state) = self.LSTM(processed_input, (hidden, cell_state))

        # eq 6 in paper
        attn_weights = self.softmax(self.sigmoid(self.attn_0(self.sigmoid(self.attn(torch.cat([decoder_context, processed_input], 2))))))
        # or encoder_outputs? i think the processed input is more intuitive
        attn_weights = attn_weights[:, :, :encoder_outputs.size(2)]

        # eq 5
        context_vector = torch.transpose(torch.matmul(torch.transpose(attn_weights.unsqueeze(2), 0, 1), torch.transpose(encoder_outputs, 0, 1)).squeeze(1), 0, 1)

        log_softmax = self.logsoftmax(self.out(torch.cat([output, context_vector], 2)))

        return log_softmax, new_hidden, attn_weights, cell_state, context_vector

    def get_initial_hidden_state(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device).cuda()
        
    def get_initial_cell_state(self, batch_size):   
        return torch.zeros(1, batch_size, self.hidden_size, device=device).cuda()

    def reset_cell_state(self, batch_size):
        return self.get_initial_cell_state(batch_size)

    def get_initial_decoder_context(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device).cuda()

######################################################################

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, args=None, max_length=MAX_LENGTH):
    batch_size = 1
    if args is not None:
        batch_size = args.batch_size

    e_hidden = encoder.get_initial_hidden_state(batch_size)
    e_cell_state = encoder.get_initial_cell_state(batch_size)
    
    optimizer.zero_grad()

    encoder.train()
    decoder.train()

    #e_out and d_in mostly similar to those in translate function
    # forward context
    e_out_list = []
    for ei in range(input_tensor.size(0)):
        eout, e_hidden, e_cell_state = encoder(input_tensor[ei], e_hidden, e_cell_state)
        e_out_list.append(eout)
    e_out_forward = torch.stack(e_out_list, dim = 2)

    e_hidden = encoder.get_initial_hidden_state(batch_size) 
    e_cell_state = encoder.get_initial_cell_state(batch_size)

    # backward context
    e_out_list = []
    for ei in range(input_tensor.size(0)):
        eout, e_hidden, e_cell_state = encoder(input_tensor[input_tensor.size(0) - ei - 1], e_hidden, e_cell_state)
        e_out_list.append(eout)
    e_out_backward = torch.stack(e_out_list, dim = 2)

    e_out = e_out_forward + e_out_backward
 
    d_in = (SOS_index * torch.ones((batch_size, 1), device=device)).int()
    
    loss = 0

    d_hidden = e_hidden
    d_hidden_last = d_hidden
    d_cell_state = decoder.get_initial_cell_state(batch_size)
    d_context = decoder.get_initial_decoder_context(batch_size)

    out_len = 0
    for i in range(target_tensor.size(0)):
        d_out, d_hidden, d_attention, d_cell_state, d_context = decoder(d_in, d_hidden, d_context, e_out, d_cell_state)
        word_loss = criterion(d_out.squeeze(0), target_tensor[i, :, 0])
        loss = loss + word_loss
        d_in = torch.transpose(torch.argmax(d_out.clone().detach(), dim = 2), 0, 1)
        d_hidden_last = d_hidden
        d_in = target_tensor[i]

    d_hidden = e_hidden
    d_hidden_last = d_hidden
    d_cell_state = decoder.get_initial_cell_state(batch_size)
    d_context = decoder.get_initial_decoder_context(batch_size)

    out_len = 0
    for i in range(target_tensor.size(0)):
        d_out, d_hidden, d_attention, d_cell_state, d_context = decoder(d_in, d_hidden, d_context, e_out, d_cell_state)
        word_loss = criterion(d_out.squeeze(0), target_tensor[i, :, 0])
        loss = loss + word_loss
        d_in = torch.transpose(torch.argmax(d_out.clone().detach(), dim = 2), 0, 1)
        d_hidden_last = d_hidden

    # in_len = target_tensor.size(0)
    # len_diff = abs(in_len - out_len)
    # loss /= out_len #average over outlen, to somewhat even out the benefit of EOS early
    # loss /= 0.9**(float(len_diff)) #punish difference in lengths
    loss.backward()
    optimizer.step()


    return loss  #can change to return what you want from this

######################################################################



######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    runs tranlsation, returns the output and attention
    """
    batch_size = 1

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_tensor = F.pad(input_tensor, (0, 0, 0, MAX_LENGTH - input_tensor.size(0)), value = 1)
        input_tensor = input_tensor.unsqueeze(1)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.get_initial_hidden_state(batch_size)
        e_cell_state = encoder.get_initial_cell_state(batch_size)

        e_out_list = []
        for ei in range(input_length):
            encoder_output, encoder_hidden, e_cell_state = encoder(input_tensor[ei],
                                                     encoder_hidden, e_cell_state)
            e_out_list.append(encoder_output)
        encoder_outputs_forward = torch.stack(e_out_list, dim = 2)
        
        encoder_hidden = encoder.get_initial_hidden_state(batch_size)
        e_cell_state = encoder.get_initial_cell_state(batch_size)

        e_out_list = []
        for ei in range(input_length):
            encoder_output, encoder_hidden, e_cell_state = encoder(input_tensor[input_length - ei - 1],
                                                     encoder_hidden, e_cell_state)
            e_out_list.append(encoder_output)
        encoder_outputs_backward = torch.stack(e_out_list, dim = 2)

        encoder_outputs = encoder_outputs_forward + encoder_outputs_backward

        decoder_input = torch.tensor([[SOS_index]], device=device)

        decoder_hidden = encoder_hidden
        d_cell_state = decoder.get_initial_cell_state(batch_size)
        decoder_context = decoder.get_initial_decoder_context(batch_size)

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, input_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention, d_cell_state, decoder_context = decoder(
                decoder_input, decoder_hidden, decoder_context, encoder_outputs, d_cell_state)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = torch.tensor([[topi.squeeze().detach()]], device=device)

        return decoded_words, decoder_attentions[:di + 1]


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, triplets, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for triplet in triplets[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, triplet[1], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, triplets, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        triplet = random.choice(triplets)
        print('>', triplet[1])
        print('=', triplet[0])
        output_words, attentions = translate(encoder, decoder, triplet[1], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions, count):
    """visualize the attention mechanism. And save it to a file. 
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """
    
    input_words = [word for word in input_sentence.split(' ')] + [EOS_token]
    tr_attentions = attentions.numpy()[:][:len(input_words)]

    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)
    axs.matshow(tr_attentions)

    axs.set_xticks(range(len(output_words)))
    axs.set_yticks(range(len(input_words)))
    axs.set_xticklabels(output_words, rotation=90)
    axs.set_yticklabels(input_words, rotation=0)
    
    axs.xaxis.set_major_locator(ticker.MultipleLocator(1))
    axs.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #fig.colorbar(axs)   axs might not be the right arg
    plt.show()
    plt.savefig('fig{}.png'.format(count))


def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab, count):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions, count)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

######################################################################

def main():

    # ******************************************************************************* #

    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=1000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--batch_size', default=64, type=int,
                    help='set batch_size')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.0005, type=int,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='en',
                    help='Source (input) language code, e.g. "jp"')
    ap.add_argument('--tgt_lang', default='jp',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--corpus_file', default='../../data/combined_with_label.txt',
                    help='training file. each line should have a source sentence,' +
                         'followed by "||", followed by a target sentence')
    # ap.add_argument('--train_file', default='data/fren.train.bpe',
    #                 help='training file. each line should have a source sentence,' +
    #                      'followed by "||", followed by a target sentence')
    # ap.add_argument('--dev_file', default='data/fren.dev.bpe',
    #                 help='dev file. each line should have a source sentence,' +
    #                      'followed by "||", followed by a target sentence')
    # ap.add_argument('--test_file', default='data/fren.test.bpe',
    #                 help='test file. each line should have a source sentence,' +
    #                      'followed by "|||", followed by a target sentence' +
    #                      ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='ENtoJP_out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')

    args = ap.parse_args()
    # args = {
    #     'hidden_size': 512,
    #     'n_iters': 200000,
    #     'print_every': 1000,
    #     'checkpoint_every': 10000,
    #     'initial_learning_rate': .0002,
    #     'src_lang': 'fr',
    #     'tgt_lang': 'en',
    #     'train_file': 'data/fren.dev.bpe',
    #     'dev_file': 'data/fren.dev.bpe',
    #     'test_file': 'data/fren.test.bpe',
    #     'out_file': 'out.txt',
    #     'load_checkpoint': None,
    #     'batch_size': 64
    # }
    # args = AttributeDict(args)


    # ******************************************************************************* #

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration 
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint)
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.corpus_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    triplets = split_lines(args.corpus_file)

    train_triplets = triplets[:8*len(triplets)//10]
    dev_triplets = triplets[8*len(triplets)//10:9*len(triplets)//10]
    test_triplets = triplets[9*len(triplets)//10:]

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every


    while iter_num < args.n_iters:
        iter_num += 1
        # print("Iteration: ", iter_num)
        training_triplet = tensors_from_triplet(src_vocab, tgt_vocab, random.choices(train_triplets, k = args.batch_size))
        input_tensor = training_triplet[0]
        target_tensor = training_triplet[1]
        loss = train(input_tensor, target_tensor, encoder,
                    decoder, optimizer, criterion, args)
        print_loss_total += loss.item()

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                    'enc_state': encoder.state_dict(),
                    'dec_state': decoder.state_dict(),
                    'opt_state': optimizer.state_dict(),
                    'src_vocab': src_vocab,
                    'tgt_vocab': tgt_vocab,
                    }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                        time.time() - start,
                        iter_num,
                        iter_num / args.n_iters * 100,
                        print_loss_avg)
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_triplets, src_vocab, tgt_vocab, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_triplets, src_vocab, tgt_vocab)

            references = [[x for x in clean(triplet[0])] for triplet in dev_triplets[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_triplets, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention
    count = 0
    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab, count)
    count += 1
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab, count)
    count += 1
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab, count)
    count += 1
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab, count)


if __name__ == '__main__':
    main()
