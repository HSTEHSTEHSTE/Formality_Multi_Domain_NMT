from transformers import AutoModel, AutoTokenizer
import torch
import model
import os
import pandas as pd
import random

bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-char")
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

# Hyper parameters
batch_size = 128

data_array = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/data.csv"))
assert data_array.shape[1] == 2
data_size = data_array.shape[0]
dev_size = int(.1 * data_size)
test_size = int(.1 * data_size)
data_array = data_array.sample(frac=1)
dev_data_array = data_array[:dev_size]
test_data_array = data_array[dev_size:dev_size + test_size]
train_data_array = data_array[dev_size + test_size:]

line = "吾辈は猫である。"

inputs = tokenizer(line, return_tensors="pt")
print(tokenizer.decode(inputs['input_ids'][0]))

outputs = bertjapanese(**inputs)
print(outputs.last_hidden_state.shape, outputs.pooler_output.shape, outputs.attentions, outputs.cross_attentions)