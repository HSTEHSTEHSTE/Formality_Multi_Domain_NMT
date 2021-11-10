import torch
from torch import nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    """Transformer encoder"""

    def __init__(self, input_size, d_model, nlayers, dropout = .5):
        self.input_size = input_size
        self.d_model = d_model
        self.nlayers = nlayers
        self.dropout = dropout
        self.embedder = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.d_model)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.nlayers)

    def forward(self, input):
        input_embedded = self.embedder(input)
        output = self.transformer_encoder(input_embedded)
        return output

class TransformerDecoder(nn.Module):
    """Transformer decoder"""

    def __init__(self, target_size, d_model, n_layers, dropout = .5):
        self.target_size = target_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.embedder = nn.Embedding(num_embeddings=self.target_size, embedding_dim=self.d_model)
        self.decoder = nn.Linear(in_features=self.d_model, out_features=self.target_size)

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.n_layers)

    def forward(self, target, memory):
        target_embedded = self.embedder(target)
        output_embedded = self.transformer_decoder(target_embedded, memory)
        output = self.decoder(output_embedded)
        return output