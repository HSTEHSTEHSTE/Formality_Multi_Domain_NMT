import torch
from torch import nn
import torch.nn.functional as F
import model
#from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.data.dictionary import Dictionary
import os
import tqdm
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import MeCab
from bpemb import BPEmb

import RIBES
import subprocess

# Hyper parameters
batch_size = 64
dev_batch_size = 64
max_iterations = 10000
test_iterations = 100
initial_learning_rate = .0005
lr_decay = .5
lr_threshold = .00001
print_every = 1000
embed_dim = 512
max_sentence_length = 15
use_gpu = True
device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu) else "cpu")
corpus_file = "data/combined_with_label_simple.txt"
corpus_file_length = 131687 # simple # 575124 total # 434407 raw # 2823 para # 575124 combined
out_file = "data/autoencoder_output_simple.txt"
translation_loss_weight = 1.

# Build config objects
#config = TransformerConfig() # default config

# Build dictionary
tokeniser = MeCab.Tagger("-Owakati")
def line_tokeniser(line):
    return tokeniser.parse(line).split()

bpemb_en = BPEmb(lang="en", dim=50)
def en_tokeniser(line):
    return bpemb_en.encode(line)

ja_dict = Dictionary()
en_dict = Dictionary()
data_file = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), corpus_file), "r", encoding="utf-8")
out_file = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), out_file), "w", encoding="utf-8")
training_triplets = []
for line in tqdm.tqdm(data_file, total=corpus_file_length):
    # Each line in data file is [JA] || [EN] || [Formality \in {0, 1}]
    elements = line.split('||')
    training_triplet = (elements[0].replace(' ', ''), elements[1].strip(), elements[2].replace('\n', ''))
    training_triplets.append(training_triplet)
    # ja_dict.encode_line(' '.join(training_triplet[0]), add_if_not_exist=True, append_eos=True)
    ja_dict.encode_line(training_triplet[0], line_tokenizer=line_tokeniser, add_if_not_exist=True)
    en_dict.encode_line(training_triplet[1], line_tokenizer=en_tokeniser, add_if_not_exist=True)

ja_dict.finalize()
en_dict.finalize()

# Build word embedding
# Use naive pytorch embedding
ja_dictionary_size = len(ja_dict)
ja_embedding = nn.Embedding(ja_dictionary_size, embed_dim, padding_idx=1) # 1: default pad value

en_dictionary_size = len(en_dict)
en_embedding = nn.Embedding(en_dictionary_size, embed_dim, padding_idx=1)

# Build encoder and decoder objects
encoder = torch.load('encoder_512.pt')
decoder = torch.load('decoder_en.pt')
encoder_back = model.TransformerEncoder(en_dictionary_size, embed_dim, n_layers=1, device=device, pad_index=en_dict.pad(), dropout=.3).to(device=device)
decoder_back = model.TransformerDecoder(ja_dictionary_size, embed_dim, n_layers=1, device=device, pad_index=ja_dict.pad(), dropout=.4).to(device=device)
# classifier = model.LinearDecoder(embed_dim * max_sentence_length, 2).to(device=device)
classifier = torch.load('classifier_512.pt')

# Load data
data_array = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), corpus_file), header=None, index_col=None, delimiter='\\|\\|').dropna()
data_size = data_array.shape[0]
dev_size = int(.1 * data_size)
test_size = int(.1 * data_size)
data_array = data_array.sample(frac=1)
dev_data_array = data_array.iloc[:dev_size]
test_data_array = data_array.iloc[dev_size:dev_size + test_size]
train_data_array = data_array.iloc[dev_size + test_size:]

# Build criterion and optimiser
criterion = torch.nn.CrossEntropyLoss(ignore_index=en_dict.pad(), label_smoothing=.3)
criterion_classifier = torch.nn.NLLLoss()
optimiser = torch.optim.Adam(list(encoder_back.parameters()) + list(decoder_back.parameters()), lr=initial_learning_rate, weight_decay=0)
softmax_decoder_output = torch.nn.LogSoftmax(dim = 2)
previous_loss = None
lr = initial_learning_rate

# Helper functions
def extract_tensors(batch_array, refs = None, ref_labels = None):
    sentence_tensors = []
    sentence_lengths = []
    formality_tensors = []
    en_sentence_tensors = []
    for sentence_label_pair in batch_array.iterrows():
        # [JA], [EN], label
        # sentence = ' '.join(sentence_label_pair[1].iloc[0])
        sentence = tokeniser.parse(sentence_label_pair[1].iloc[0].replace(' ', ''))
        sentence_tensor = ja_dict.encode_line(sentence, append_eos=True) # (len(sentence) + 1)
        if sentence_tensor.shape[0] <= max_sentence_length:
            if refs is not None:
                refs.append([sentence.split()])
            if ref_labels is not None:
                ref_labels.append(int(sentence_label_pair[1].iloc[2]))
            sentence_tensor = torch.cat([torch.tensor([ja_dict.bos()]), sentence_tensor])
            sentence_lengths.append(min(max_sentence_length, sentence_tensor.shape[0]))
            sentence_tensors.append(F.pad(sentence_tensor, (0, max_sentence_length - sentence_tensor.shape[0]), value = ja_dict.pad())[:max_sentence_length]) # (max_sentence_length)

            en_sentence = ' '.join(en_tokeniser(sentence_label_pair[1].iloc[1]))
            en_sentence_tensor = en_dict.encode_line(en_sentence, append_eos=True)
            en_sentence_tensor = torch.cat([torch.tensor([en_dict.bos()]), en_sentence_tensor])
            en_sentence_tensors.append(F.pad(en_sentence_tensor, (0, max_sentence_length - en_sentence_tensor.shape[0]), value = en_dict.pad())[:max_sentence_length]) # (max_sentence_length)

            formality_tensors.append(int(sentence_label_pair[1].iloc[2]))

    sentence_tensors = torch.stack(sentence_tensors, dim=0).long().to(device=device) # (batch_size, max_sentence_length + 1)
    en_sentence_tensors = torch.stack(en_sentence_tensors, dim=0).long().to(device=device) # (batch_size, max_sentence_length + 1)
    sentence_lengths = torch.tensor(sentence_lengths).to(device=device) # (batch_size)
    formality_tensors = torch.tensor(formality_tensors).to(device=device).long() # (batch_size)

    return sentence_tensors, sentence_lengths, formality_tensors, en_sentence_tensors

total_loss = 0
total_dev_loss = 0
for iteration_number in range(0, max_iterations):
    # load data
    batch_array = train_data_array.sample(n=batch_size)
    sentence_tensors, sentence_lengths, formality_tensors, en_sentence_tensors = extract_tensors(batch_array)

    encoder_back.train()
    decoder_back.train()
    classifier.train()

    optimiser.zero_grad()

    # main forward pass
    loss = torch.tensor(0.).to(device=device)
    memory, pad_mask = encoder(sentence_tensors)
    classifier_output = classifier((memory * (-1 * pad_mask.float() + 1).unsqueeze(2)).view(sentence_tensors.shape[0], -1))

    # teacher forcing
    memory, pad_mask = encoder_back(en_sentence_tensors)
    decoder_output = softmax_decoder_output(decoder_back(sentence_tensors[:, :-1], memory, pad_mask))
    loss = translation_loss_weight * criterion(decoder_output.view(-1, decoder_output.shape[2]), sentence_tensors[:, 1:].reshape(-1).long())
    memory, pad_mask = encoder(torch.cat([sentence_tensors[:, 0].unsqueeze(1), torch.argmax(decoder_output, dim=2)], dim=1))
    classifier_output = classifier((memory * (-1 * pad_mask.float() + 1).unsqueeze(2)).view(sentence_tensors.shape[0], -1))
    loss += criterion_classifier(classifier_output, formality_tensors)

    total_loss += loss.item()
    loss.backward()
    optimiser.step()

    # load dev data
    dev_batch_array = dev_data_array.sample(n=dev_batch_size)
    refs = []
    dev_sentence_tensors, dev_sentence_lengths, dev_formality_tensors, dev_en_sentence_tensors = extract_tensors(dev_batch_array, refs)

    encoder_back.eval()
    decoder_back.eval()
    classifier.eval()

    optimiser.zero_grad()

    # dev forward pass
    loss = torch.tensor(0.).to(device=device)
    memory, pad_mask = encoder(dev_sentence_tensors)
    dev_classifier_output = classifier((memory * (-1 * pad_mask.float() + 1).unsqueeze(2)).view(dev_sentence_tensors.shape[0], -1))
    decoder_output = torch.tensor(en_dict.bos()).unsqueeze(0).long().repeat(dev_sentence_tensors.shape[0], 1).to(device=device) # (batch_size, 1)
    has_reached_eos = torch.ones([dev_sentence_tensors.shape[0], 1]).to(device=device) # (batch_size, 1)
    eoses = torch.tensor(en_dict.eos()).unsqueeze(0).long().repeat(dev_sentence_tensors.shape[0], 1).to(device=device) # (batch_size, 1)
    for output_index in range(1, max_sentence_length):
        next_output = softmax_decoder_output(decoder(decoder_output, memory, pad_mask))
        # has_reached_eos = has_reached_eos * ((torch.argmax(next_output[:, -1, :].unsqueeze(1), dim = 2) - eoses) > .5)
        has_reached_eos = has_reached_eos * ((dev_en_sentence_tensors[:, output_index].unsqueeze(1) - eoses) > .5)
        decoder_output = torch.cat([decoder_output.detach(), torch.argmax(next_output[:, -1, :].detach().unsqueeze(1), dim=2)], dim=1)

    # dev back translation
    memory, pad_mask = encoder_back(decoder_output)
    decoder_output = torch.tensor(en_dict.bos()).unsqueeze(0).long().repeat(dev_sentence_tensors.shape[0], 1).to(device=device) # (batch_size, 1)
    has_reached_eos = torch.ones([dev_sentence_tensors.shape[0], 1]).to(device=device) # (batch_size, 1)
    eoses = torch.tensor(en_dict.eos()).unsqueeze(0).long().repeat(dev_sentence_tensors.shape[0], 1).to(device=device) # (batch_size, 1)
    for output_index in range(1, max_sentence_length):
        next_output = softmax_decoder_output(decoder_back(decoder_output, memory, pad_mask))
        # has_reached_eos = has_reached_eos * ((torch.argmax(next_output[:, -1, :].unsqueeze(1), dim = 2) - eoses) > .5)
        has_reached_eos = has_reached_eos * ((dev_sentence_tensors[:, output_index].unsqueeze(1) - eoses) > .5)
        loss += translation_loss_weight * criterion(next_output[:, -1, :], dev_sentence_tensors[:, output_index].long())
        decoder_output = torch.cat([decoder_output.detach(), torch.argmax(next_output[:, -1, :].detach().unsqueeze(1), dim=2)], dim=1)
    memory, pad_mask = encoder(decoder_output)
    dev_classifier_output = classifier((memory * (-1 * pad_mask.float() + 1).unsqueeze(2)).view(dev_sentence_tensors.shape[0], -1))
    loss += criterion_classifier(dev_classifier_output, dev_formality_tensors)

    total_dev_loss += loss.item()

    # calculate dev loss, update learning rate
    if (iteration_number + 1) % print_every == 0:
        print("Iteration ", iteration_number + 1, " , loss is ", total_loss / print_every)
        total_loss = 0
        
        dev_loss = total_dev_loss / print_every
        print("Dev loss is ", dev_loss)
        total_dev_loss = 0
        
        # # update learning rate
        # if previous_loss is not None and previous_loss < dev_loss:
        #     lr_new = lr * lr_decay
        #     print("Dev loss increased. Reducing learning rate from ", lr, " to ", lr_new)
        #     lr = lr_new
        #     for param_group in optimiser.param_groups:
        #         param_group["lr"] = lr
        # previous_loss = dev_loss

        # calculate dev bleu score
        formality = torch.argmax(dev_classifier_output, dim=1)
        formality_accuracy = torch.div(torch.sum((formality == dev_formality_tensors).float()), formality.shape[0])
        print("Dev formality accuracy: ", formality_accuracy.item())

        hyps = []
        for index, dev_sentence in enumerate(decoder_output):
            sentence_characters = ja_dict.string(dev_sentence)
            hyps.append(sentence_characters.split())

            print(hyps[index])
            print(refs[index][0])
        
        print(nltk.translate.bleu_score.corpus_bleu(refs, hyps, weights=(.34, .33, .33)))
        print(nltk.translate.ribes_score.corpus_ribes(refs, hyps))
        if lr < lr_threshold:
            break

# do one pass over the test corpus
refs = []
ref_labels = []
hyps = []
total_test_loss = 0.
for iteration_number in tqdm.tqdm(range(0, test_iterations), total=test_iterations):
    # load test data
    test_batch_array = test_data_array.sample(n=dev_batch_size)
    test_sentence_tensors, test_sentence_lengths, test_formality_tensors, test_en_sentence_tensors = extract_tensors(batch_array, refs, ref_labels)

    encoder_back.eval()
    decoder_back.eval()
    classifier.eval()

    optimiser.zero_grad()

    # test forward pass
    loss = torch.tensor(0.).to(device=device)
    memory, pad_mask = encoder(test_sentence_tensors)
    test_classifier_output = classifier((memory * (-1 * pad_mask.float() + 1).unsqueeze(2)).view(test_sentence_tensors.shape[0], -1))
    decoder_output = torch.tensor(en_dict.bos()).unsqueeze(0).long().repeat(test_sentence_tensors.shape[0], 1).to(device=device) # (batch_size, 1)
    has_reached_eos = torch.ones([test_sentence_tensors.shape[0], 1]).to(device=device) # (batch_size, 1)
    eoses = torch.tensor(ja_dict.eos()).unsqueeze(0).long().repeat(test_sentence_tensors.shape[0], 1).to(device=device) # (batch_size, 1)
    for output_index in range(1, max_sentence_length):
        next_output = softmax_decoder_output(decoder(decoder_output, memory, pad_mask))
        # has_reached_eos = has_reached_eos * ((torch.argmax(next_output[:, -1, :].unsqueeze(1), dim = 2) - eoses) > .5)
        has_reached_eos = has_reached_eos * ((test_en_sentence_tensors[:, output_index].unsqueeze(1) - eoses) > .5)
        loss += translation_loss_weight * criterion(next_output[:, -1, :], test_en_sentence_tensors[:, output_index].long())
        decoder_output = torch.cat([decoder_output.detach(), torch.argmax(next_output[:, -1, :].detach().unsqueeze(1), dim=2)], dim=1)

    # test back translation
    memory, pad_mask = encoder_back(decoder_output)
    decoder_output = torch.tensor(en_dict.bos()).unsqueeze(0).long().repeat(test_sentence_tensors.shape[0], 1).to(device=device) # (batch_size, 1)
    has_reached_eos = torch.ones([test_sentence_tensors.shape[0], 1]).to(device=device) # (batch_size, 1)
    eoses = torch.tensor(en_dict.eos()).unsqueeze(0).long().repeat(test_sentence_tensors.shape[0], 1).to(device=device) # (batch_size, 1)
    for output_index in range(1, max_sentence_length):
        next_output = softmax_decoder_output(decoder_back(decoder_output, memory, pad_mask))
        # has_reached_eos = has_reached_eos * ((torch.argmax(next_output[:, -1, :].unsqueeze(1), dim = 2) - eoses) > .5)
        has_reached_eos = has_reached_eos * ((test_sentence_tensors[:, output_index].unsqueeze(1) - eoses) > .5)
        loss += translation_loss_weight * criterion(next_output[:, -1, :], test_sentence_tensors[:, output_index].long())
        decoder_output = torch.cat([decoder_output.detach(), torch.argmax(next_output[:, -1, :].detach().unsqueeze(1), dim=2)], dim=1)
    memory, pad_mask = encoder(decoder_output)
    test_classifier_output = classifier((memory * (-1 * pad_mask.float() + 1).unsqueeze(2)).view(test_sentence_tensors.shape[0], -1))
    loss += criterion_classifier(test_classifier_output, test_formality_tensors)

    total_test_loss += loss.item()

    # Log output sentences
    for index, test_sentence in enumerate(decoder_output):
        sentence_characters = ja_dict.string(test_sentence)
        hyps.append(sentence_characters.split())

for index, hyp in enumerate(hyps):
    write_line = ''.join(hyp) + ' || ' + ''.join(refs[index][0]) + ' || ' + str(ref_labels[index])
    out_file.write(write_line + '\n')

# save trained models
torch.save(encoder_back, 'encoder_back.pt')
torch.save(decoder_back, 'decoder_back.pt')

test_loss = total_test_loss / test_iterations
print("Test loss is ", test_loss)

# Calculate BLEU score
print("Test BLEU score: ", nltk.translate.bleu_score.corpus_bleu(refs, hyps, weights=(.34, .33, .33)))
print("Test RIBES score: ", nltk.translate.ribes_score.corpus_ribes(refs, hyps))
