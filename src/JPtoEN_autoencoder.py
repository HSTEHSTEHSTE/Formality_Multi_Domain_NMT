import torch
from torch import nn
import torch.nn.functional as F
import model
from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.data.dictionary import Dictionary
import os
import tqdm
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import MeCab

# Hyper parameters
batch_size = 8
dev_batch_size = 8
max_iterations = 10000
test_iterations = 1000
initial_learning_rate = .001
lr_decay = .5
lr_threshold = .00001
print_every = 10
embed_dim = 256
max_sentence_length = 15
use_gpu = True
device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu) else "cpu")

# Build config objects
config = TransformerConfig() # default config

# Build dictionary
tokeniser = MeCab.Tagger("-Owakati")
def line_tokeniser(line):
    return tokeniser.parse(line).split()

en_tokeniser = word_tokenize
def en_line_tokeniser(line):
    return en_tokeniser(line)  

ja_dict = Dictionary()
en_dict = Dictionary()
data_file = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/combined_with_label.txt"), "r", encoding="utf-8")
training_triplets = []
for line in tqdm.tqdm(data_file, total=575124):
    # Each line in data file is [JA] || [EN] || [Formality \in {0, 1}]
    elements = line.split('||')
    training_triplet = (elements[0].replace(' ', ''), elements[1].strip(), elements[2].replace('\n', ''))
    training_triplets.append(training_triplet)
    # ja_dict.encode_line(' '.join(training_triplet[0]), add_if_not_exist=True, append_eos=True)
    ja_dict.encode_line(training_triplet[0], line_tokenizer=line_tokeniser, add_if_not_exist=True)
    en_dict.encode_line(training_triplet[1], line_tokenizer=en_line_tokeniser, add_if_not_exist=True)

ja_dict.finalize()
en_dict.finalize()

# Build word embedding
# Use naive pytorch embedding
ja_dictionary_size = len(ja_dict)
ja_embedding = nn.Embedding(ja_dictionary_size, embed_dim, padding_idx=1) # 1: default pad value

en_dictionary_size = len(en_dict)
en_embedding = nn.Embedding(en_dictionary_size, embed_dim, padding_idx=1)

# Build encoder and decoder objects
encoder = model.TransformerEncoder(ja_dictionary_size, embed_dim, 1, device=device, pad_index=ja_dict.pad()).to(device=device)
decoder = model.TransformerDecoder(en_dictionary_size, embed_dim, 1, device=device, pad_index=en_dict.pad(), dropout=.2).to(device=device)

# Load data
data_array = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/combined_with_label.txt"), header=None, index_col=None, delimiter='\\|\\|').dropna()
data_size = data_array.shape[0]
dev_size = int(.1 * data_size)
test_size = int(.1 * data_size)
data_array = data_array.sample(frac=1)
dev_data_array = data_array.iloc[:dev_size]
test_data_array = data_array.iloc[dev_size:dev_size + test_size]
train_data_array = data_array.iloc[dev_size + test_size:]

# Build criterion and optimiser
criterion = torch.nn.NLLLoss(ignore_index=en_dict.pad())
optimiser = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=initial_learning_rate, weight_decay=0)
softmax_decoder_output = torch.nn.LogSoftmax(dim = 2)
previous_loss = None
lr = initial_learning_rate

total_loss = 0
total_dev_loss = 0
for iteration_number in range(0, max_iterations):
    batch_array = train_data_array.sample(n=batch_size)

    sentence_tensors = []
    #sentence_lengths = []
    en_sentence_tensors = []
    for sentence_label_pair in batch_array.iterrows():
        # [JA], [EN], label
        # sentence = ' '.join(sentence_label_pair[1].iloc[0])
        sentence = tokeniser.parse(sentence_label_pair[1].iloc[0].replace(' ', ''))
        sentence_tensor = ja_dict.encode_line(sentence, append_eos=True) # (len(sentence) + 1)
        sentence_tensor = torch.cat([torch.tensor([ja_dict.bos()]), sentence_tensor])
        #en_sentence = [x for x in sentence_label_pair[1].iloc[1].split()]
        en_sentence = ' '.join(en_tokeniser(sentence_label_pair[1].iloc[1]))
        en_sentence_tensor = en_dict.encode_line(en_sentence, append_eos=True)
        en_sentence_tensor = torch.cat([torch.tensor([en_dict.bos()]), en_sentence_tensor])

        #sentence_lengths.append(min(max_sentence_length, sentence_tensor.shape[0]))
        sentence_tensors.append(F.pad(sentence_tensor, (0, max_sentence_length - sentence_tensor.shape[0]), value = ja_dict.pad())[:max_sentence_length]) # (max_sentence_length)
        en_sentence_tensors.append(F.pad(en_sentence_tensor, (0, max_sentence_length - en_sentence_tensor.shape[0]), value = en_dict.pad())[:max_sentence_length]) # (max_sentence_length)

    sentence_tensors = torch.stack(sentence_tensors, dim=0).long().to(device=device) # (batch_size, max_sentence_length + 1)
    en_sentence_tensors = torch.stack(en_sentence_tensors, dim=0).long().to(device=device) # (batch_size, max_sentence_length + 1)
    #sentence_lengths = torch.tensor(sentence_lengths).to(device=device) # (batch_size)

    encoder.train()
    decoder.train()
    optimiser.zero_grad()

    # main forward pass
    loss = torch.tensor(0.).to(device=device)
    memory, pad_mask = encoder(sentence_tensors)

    # teacher forcing
    decoder_output = softmax_decoder_output(decoder(en_sentence_tensors[:, :-1], memory, pad_mask))
    loss = criterion(decoder_output.view(-1, decoder_output.shape[2]), en_sentence_tensors[:, 1:].reshape(-1).long())


    total_loss += loss.item()
    loss.backward()
    optimiser.step()

    # load dev data
    dev_batch_array = dev_data_array.sample(n=dev_batch_size)
    dev_sentence_tensors = []
    #dev_sentence_lengths = []
    dev_en_sentence_tensors = []
    refs = []
    for sentence_label_pair in dev_batch_array.iterrows():
        # [JA], [EN], label
        refs.append([en_tokeniser(sentence_label_pair[1].iloc[1])])

        dev_sentence = tokeniser.parse(sentence_label_pair[1].iloc[0].replace(' ', ''))
        dev_sentence_tensor = ja_dict.encode_line(dev_sentence, append_eos=True) # (len(sentence) + 1)
        dev_sentence_tensor = torch.cat([torch.tensor([ja_dict.bos()]), dev_sentence_tensor])
        #dev_sentence_lengths.append(min(max_sentence_length, dev_sentence_tensor.shape[0]))
        dev_sentence_tensors.append(F.pad(dev_sentence_tensor, (0, max_sentence_length - dev_sentence_tensor.shape[0]), value = ja_dict.pad())[:max_sentence_length]) # (max_sentence_length)

        dev_en_sentence = ' '.join(en_tokeniser(sentence_label_pair[1].iloc[1]))
        dev_en_sentence_tensor = en_dict.encode_line(dev_en_sentence, append_eos=True) # (len(sentence) + 1)
        dev_en_sentence_tensor = torch.cat([torch.tensor([en_dict.bos()]), dev_en_sentence_tensor])
        dev_en_sentence_tensors.append(F.pad(dev_en_sentence_tensor, (0, max_sentence_length - dev_en_sentence_tensor.shape[0]), value = en_dict.pad())[:max_sentence_length]) # (max_sentence_length)
    
    dev_sentence_tensors = torch.stack(dev_sentence_tensors, dim=0).long().to(device=device) # (batch_size, max_sentence_length)
    dev_en_sentence_tensors = torch.stack(dev_en_sentence_tensors, dim=0).long().to(device=device) # (batch_size, max_sentence_length)
    #dev_sentence_lengths = torch.tensor(dev_sentence_lengths).to(device=device) # (batch_size)

    encoder.eval()
    decoder.eval()

    optimiser.zero_grad()

    # dev forward pass
    loss = torch.tensor(0.).to(device=device)
    memory, pad_mask = encoder(dev_sentence_tensors)
    decoder_output = torch.tensor(en_dict.bos()).unsqueeze(0).long().repeat(batch_size, 1).to(device=device) # (batch_size, 1)
    #print(decoder_output)
    has_reached_eos = torch.ones([batch_size, 1]).to(device=device) # (batch_size, 1)
    eoses = torch.tensor(en_dict.eos()).unsqueeze(0).long().repeat(batch_size, 1).to(device=device) # (batch_size, 1)
    for output_index in range(1, max_sentence_length - 1):
        next_output = softmax_decoder_output(decoder(decoder_output, memory, pad_mask))
        # has_reached_eos = has_reached_eos * ((torch.argmax(next_output[:, -1, :].unsqueeze(1), dim = 2) - eoses) > .5)
        has_reached_eos = has_reached_eos * ((dev_en_sentence_tensors[:, output_index].unsqueeze(1) - eoses) > .5)
        loss += criterion(next_output[:, -1, :], dev_en_sentence_tensors[:, output_index].long())
        decoder_output = torch.cat([decoder_output.detach(), torch.argmax(next_output[:, -1, :].detach().unsqueeze(1), dim=2)], dim=1)
    total_dev_loss += loss.item()
    #print(decoder_output)

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
        hyps = []
        for index, dev_sentence in enumerate(decoder_output):
            sentence_characters = en_dict.string(dev_sentence)
            #print(sentence_characters)
            hyps.append(sentence_characters.split())

            # for char in dev_sentence:
            #     print(ja_dict.symbols[torch.argmax(char)])
            print(hyps[index])
            print(refs[index][0])
        
        print(nltk.translate.bleu_score.corpus_bleu(refs, hyps))

        if lr < lr_threshold:
            break

# do one pass over the test corpus

refs = []
hyps = []
for iteration_number in tqdm.tqdm(range(0, test_iterations), total=test_iterations):
    # load test data
    test_batch_array = test_data_array.sample(n=dev_batch_size)
    test_sentence_tensors = []
    #test_sentence_lengths = []
    test_en_sentence_tensors = []
    total_test_loss = 0.
    for sentence_label_pair in test_batch_array.iterrows():
        # [JA], [EN], label
        # test_sentence = ' '.join(sentence_label_pair[1].iloc[0])
        test_sentence = tokeniser.parse(sentence_label_pair[1].iloc[0].replace(' ', ''))
        refs.append([en_tokeniser(sentence_label_pair[1].iloc[1])])
        test_sentence_tensor = ja_dict.encode_line(test_sentence, append_eos=True) # (len(sentence) + 1)
        test_sentence_tensor = torch.cat([torch.tensor([ja_dict.bos()]), test_sentence_tensor])
        #test_sentence_lengths.append(min(max_sentence_length, test_sentence_tensor.shape[0]))
        test_sentence_tensors.append(F.pad(test_sentence_tensor, (0, max_sentence_length - test_sentence_tensor.shape[0]), value = ja_dict.pad())[:max_sentence_length]) # (max_sentence_length)

        test_en_sentence = ' '.join(en_tokeniser(sentence_label_pair[1].iloc[1]))
        test_en_sentence_tensor = en_dict.encode_line(test_en_sentence, append_eos=True) # (len(sentence) + 1)
        test_en_sentence_tensor = torch.cat([torch.tensor([en_dict.bos()]), test_en_sentence_tensor])
        test_en_sentence_tensors.append(F.pad(test_en_sentence_tensor, (0, max_sentence_length - test_en_sentence_tensor.shape[0]), value = en_dict.pad())[:max_sentence_length]) # (max_sentence_length)

    test_sentence_tensors = torch.stack(test_sentence_tensors, dim=0).long().to(device=device) # (batch_size, max_sentence_length)
    test_en_sentence_tensors = torch.stack(test_en_sentence_tensors, dim=0).long().to(device=device) # (batch_size, max_sentence_length)
    #test_sentence_lengths = torch.tensor(test_sentence_lengths).to(device=device) # (batch_size)

    encoder.eval()
    decoder.eval()

    optimiser.zero_grad()

    # test forward pass
    loss = torch.tensor(0.).to(device=device)
    memory, pad_mask = encoder(test_sentence_tensors)
    decoder_output = torch.tensor(en_dict.bos()).unsqueeze(0).long().repeat(batch_size, 1).to(device=device) # (batch_size, 1)
    has_reached_eos = torch.ones([batch_size, 1]).to(device=device) # (batch_size, 1)
    eoses = torch.tensor(ja_dict.eos()).unsqueeze(0).long().repeat(batch_size, 1).to(device=device) # (batch_size, 1)
    for output_index in range(1, max_sentence_length - 1):
        next_output = softmax_decoder_output(decoder(decoder_output, memory, pad_mask))
        # has_reached_eos = has_reached_eos * ((torch.argmax(next_output[:, -1, :].unsqueeze(1), dim = 2) - eoses) > .5)
        has_reached_eos = has_reached_eos * ((test_en_sentence_tensors[:, output_index].unsqueeze(1) - eoses) > .5)
        loss += criterion(next_output[:, -1, :], test_en_sentence_tensors[:, output_index].long())
        decoder_output = torch.cat([decoder_output.detach(), torch.argmax(next_output[:, -1, :].detach().unsqueeze(1), dim=2)], dim=1)
    total_test_loss += loss.item()

    # Log output sentences
    for index, test_sentence in enumerate(decoder_output):
        sentence_characters = en_dict.string(test_sentence)
        hyps.append(sentence_characters.split())

test_loss = total_test_loss / (test_iterations * dev_batch_size)
print("Test loss is ", test_loss)

# Calculate BLEU score
print("Test BLEU score: ", nltk.translate.bleu_score.corpus_bleu(refs, hyps))
