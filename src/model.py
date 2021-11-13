import torch
from torch import nn
import torch.nn.functional as F
import math

class TransformerEncoder(nn.Module):
    """Transformer encoder"""

    def __init__(self, input_size, d_model, nlayers, dropout = .5):
        super(TransformerEncoder, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.nlayers = nlayers
        self.dropout = dropout
        self.embedder = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.d_model)
        self.position_encoder = PositionalEncoding(d_model=self.d_model, dropout=self.dropout)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.nlayers)

    def forward(self, input):
        input_embedded = self.embedder(input)
        input_embedded = self.position_encoder(input_embedded)
        output = self.transformer_encoder(input_embedded)
        return output

class TransformerDecoder(nn.Module):
    """Transformer decoder"""

    def __init__(self, target_size, d_model, n_layers, dropout = .5):
        super(TransformerDecoder, self).__init__()
        self.target_size = target_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedder = nn.Embedding(num_embeddings=self.target_size, embedding_dim=self.d_model)
        self.position_encoder = PositionalEncoding(d_model=self.d_model, dropout=self.dropout)
        self.decoder = nn.Linear(in_features=self.d_model, out_features=self.target_size)

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, dropout=self.dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.n_layers)

    def generate_mask(self):
        return torch.triu(torch.ones(self.d_model, self.d_model) * float('-inf'), diagonal=1)

    def forward(self, target, memory):
        target_mask = self.generate_mask()
        target_embedded = self.embedder(target)
        target_embedded = self.position_encoder(target_embedded)
        output_embedded = self.transformer_decoder(target_embedded, memory, tgt_mask=target_mask)
        output = self.decoder(output_embedded)
        return output

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LinearDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearDecoder, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=2)
    
    def forward(self, classifier_input):
        """
            input: [batch_size, input_size]
        """
        linear_output = self.linear(classifier_input)
        logsoftmax_output = self.logsoftmax(linear_output)
        return logsoftmax_output