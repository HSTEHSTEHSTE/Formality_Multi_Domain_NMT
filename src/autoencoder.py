import torch
from torch import nn
import torch.nn.functional as F
from fairseq.models.transformer.transformer_encoder import TransformerEncoder
from fairseq.models.transformer.transformer_decoder import TransformerDecoder
from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.data.dictionary import Dictionary
import os
import tqdm
import pandas as pd

# Hyper parameters
batch_size = 64
dev_batch_size = 64
max_iterations = 1000
initial_learning_rate = .001
lr_decay = .9
lr_threshold = .00001
print_every = 2
max_sentence_length = 20
use_gpu = True
device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu) else "cpu")

# Build config objects
config = TransformerConfig() # default config

# Build dictionary
ja_dict = Dictionary()
data_file = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/combined_with_label.txt"), "r", encoding="utf-8")
training_triplets = []
for line in tqdm.tqdm(data_file, total=575124):
    # Each line in data file is [JA] || [EN] || [Formality \in {0, 1}]
    elements = line.split('||')
    training_triplet = (elements[0].strip(), elements[1].strip(), elements[2].replace('\n', ''))
    training_triplets.append(training_triplet)
    ja_dict.encode_line(' '.join(training_triplet[0]), add_if_not_exist=True)

ja_dict.finalize()

# Build word embedding
# Use naive pytorch embedding
ja_dictionary_size = len(ja_dict)
ja_embedding = nn.Embedding(ja_dictionary_size, 512, padding_idx=1) # 512: default embed_dim; 1: default pad value

# Build encoder and decoder objects
encoder = TransformerEncoder(config, ja_dict, ja_embedding).to(device=device)
decoder = TransformerDecoder(config, ja_dict, ja_embedding).to(device=device)

# Load data
data_array = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/combined_with_label.txt"), header=None, index_col=None, delimiter='\\|\\|').dropna()
data_size = data_array.shape[0]
dev_size = int(.1 * data_size)
test_size = int(.1 * data_size)
data_array = data_array.sample(frac=1)
dev_data_array = data_array.loc[:dev_size]
test_data_array = data_array.loc[dev_size:dev_size + test_size]
train_data_array = data_array.loc[dev_size + test_size:]

# Build criterion and optimiser
criterion = torch.nn.NLLLoss()
optimiser = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=initial_learning_rate, weight_decay=0)
softmax_decoder_output = torch.nn.LogSoftmax(dim = 2)
previous_loss = None
lr = initial_learning_rate

total_loss = 0
total_dev_loss = 0
for iteration_number in range(0, max_iterations):
    batch_array = train_data_array.sample(n=batch_size)

    sentence_tensors = []
    sentence_lengths = []
    for sentence_label_pair in batch_array.iterrows():
        # [JA], [EN], label
        sentence = ' '.join(sentence_label_pair[1].iloc[0])
        sentence_tensor = ja_dict.encode_line(sentence) # (len(sentence) + 1)
        sentence_lengths.append(min(max_sentence_length, sentence_tensor.shape[0]))
        sentence_tensors.append(F.pad(sentence_tensor, (0, max_sentence_length - sentence_tensor.shape[0]), value = ja_dict.pad())[:max_sentence_length]) # (max_sentence_length)

    sentence_tensors = torch.stack(sentence_tensors, dim=0).to(device=device) # (batch_size, max_sentence_length)
    sentence_lengths = torch.tensor(sentence_lengths).to(device=device) # (batch_size)

    encoder.train()
    decoder.train()
    optimiser.zero_grad()

    # main forward pass
    encoder_dict = encoder(sentence_tensors, sentence_lengths)
    decoder_output = softmax_decoder_output(decoder(sentence_tensors, encoder_dict)[0])
    loss = criterion(decoder_output.view(-1, decoder_output.shape[2]), sentence_tensors.view(-1).long())
    loss.backward()
    optimiser.step()
    total_loss += loss.item()

    # load dev data
    dev_batch_array = dev_data_array.sample(n=dev_batch_size)
    dev_sentence_tensors = []
    dev_sentence_lengths = []
    for sentence_label_pair in dev_batch_array.iterrows():
        # [JA], [EN], label
        dev_sentence = ' '.join(sentence_label_pair[1].iloc[0])
        dev_sentence_tensor = ja_dict.encode_line(dev_sentence) # (len(sentence) + 1)
        dev_sentence_lengths.append(min(max_sentence_length, dev_sentence_tensor.shape[0]))
        dev_sentence_tensors.append(F.pad(dev_sentence_tensor, (0, max_sentence_length - dev_sentence_tensor.shape[0]), value = ja_dict.pad())[:max_sentence_length]) # (max_sentence_length)
    
    dev_sentence_tensors = torch.stack(dev_sentence_tensors, dim=0).to(device=device) # (batch_size, max_sentence_length)
    dev_sentence_lengths = torch.tensor(dev_sentence_lengths).to(device=device) # (batch_size)

    encoder.eval()
    decoder.eval()

    optimiser.zero_grad()

    # dev forward pass
    encoder_dict = encoder(dev_sentence_tensors, dev_sentence_lengths)
    decoder_output = softmax_decoder_output(decoder(dev_sentence_tensors, encoder_dict)[0])
    loss = criterion(decoder_output.view(-1, decoder_output.shape[2]), sentence_tensors.view(-1).long())

    total_dev_loss += loss.item()


    # calculate dev loss, update learning rate
    if (iteration_number + 1) % print_every == 0:
        print("Iteration ", iteration_number + 1, " , loss is ", total_loss / print_every)
        total_loss = 0
        
        dev_loss = total_dev_loss / print_every
        print("Dev loss is ", dev_loss)
        total_dev_loss = 0
        
        # update learning rate
        if previous_loss is not None and previous_loss < dev_loss:
            lr_new = lr * lr_decay
            print("Dev loss increased. Reducing learning rate from ", lr, " to ", lr_new)
            lr = lr_new
            for param_group in optimiser.param_groups:
                param_group["lr"] = lr
        previous_loss = dev_loss

        if lr < lr_threshold:
            break