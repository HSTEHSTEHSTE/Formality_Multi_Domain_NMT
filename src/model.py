import torch
from torch import nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    """Transfomer encoder"""

    def __init__(self, input_length, d_model, nlayers, dropout = .5):
        self.input_length = input_length
        self.d_model = d_model
        self.nlayers = nlayers
        self.dropout = dropout
        self.embedder = nn.Embedding(num_embeddings=self.input_length, embedding_dim=self.d_model)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.nlayers)

    def forward(self, input):
        input_embedded = self.embedder(input)
        output = self.transformer_encoder(input_embedded)
        return output