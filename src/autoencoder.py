import torch
from torch import nn
from fairseq.models.transformer.transformer_encoder import TransformerEncoder
from fairseq.models.transformer.transformer_decoder import TransformerDecoder
from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.data.dictionary import Dictionary
import os
import tqdm
import pandas as pd

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
    ja_dict.encode_line(training_triplet[0], add_if_not_exist=True)

ja_dict.finalize()

# Build word embedding
# Use naive pytorch embedding
ja_dictionary_size = len(ja_dict)
ja_embedding = nn.Embedding(ja_dictionary_size, 512, padding_idx=1) # 512: default embed_dim; 1: default pad value

# Build encoder and decoder objects
encoder = TransformerEncoder(config, ja_dict, ja_embedding)
decoder = TransformerDecoder(config, ja_dict, ja_embedding)

data_array = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/combined_with_label.txt"), header=None, index_col=None, delimiter='\\|\\|')
data_size = data_array.shape[0]
dev_size = int(.1 * data_size)
test_size = int(.1 * data_size)
data_array = data_array.sample(frac=1)
dev_data_array = data_array.loc[:dev_size]
test_data_array = data_array.loc[dev_size:dev_size + test_size]
train_data_array = data_array.loc[dev_size + test_size:]