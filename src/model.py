import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
import math
import numpy as np

class TransformerEncoder(nn.Module):
    """Transformer encoder"""

    def __init__(self, input_size, d_model, nlayers, device, pad_index = 1, dropout = .1):
        super(TransformerEncoder, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.nlayers = nlayers
        self.pad_index = pad_index
        self.dropout = dropout
        self.embedder = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.d_model)
        self.position_encoder = PositionalEncoding(d_model=self.d_model, dropout=self.dropout, device=device)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, dropout=self.dropout, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.nlayers)

    def generate_pad_mask(self, input):
        return (torch.abs(input - self.pad_index) < .5)

    def forward(self, input_tensor):
        input_embedded_pre_pos = self.embedder(input_tensor).float()
        input_pad_mask = self.generate_pad_mask(input_tensor)
        input_embedded = self.position_encoder(input_embedded_pre_pos)
        output = self.transformer_encoder(input_embedded, src_key_padding_mask=input_pad_mask)
        return output, input_pad_mask

class TransformerDecoder(nn.Module):
    """Transformer decoder"""

    def __init__(self, target_size, d_model, n_layers, device, pad_index = 1, dropout = .1):
        super(TransformerDecoder, self).__init__()
        self.target_size = target_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.pad_index = pad_index
        self.dropout = dropout
        self.device = device
        self.embedder = nn.Embedding(num_embeddings=self.target_size, embedding_dim=self.d_model)
        self.position_encoder = PositionalEncoding(d_model=self.d_model, dropout=self.dropout, device=device)
        self.decoder = nn.Linear(in_features=self.d_model, out_features=self.target_size)

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, dropout=self.dropout, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.n_layers)

    def generate_mask(self, size):
        mask = nn.Transformer.generate_square_subsequent_mask(size).to(device=self.device)
        return mask

    def generate_pad_mask(self, input):
        return (torch.abs(input - self.pad_index) < .5)

    def forward(self, target, memory, memory_pad_mask):
        target_mask = self.generate_mask(target.shape[1])
        target_pad_mask = self.generate_pad_mask(target)
        target_embedded = self.embedder(target).float()
        target_embedded = self.position_encoder(target_embedded)
        output_embedded = self.transformer_decoder(target_embedded, memory, tgt_mask=target_mask, tgt_key_padding_mask=target_pad_mask, memory_key_padding_mask=memory_pad_mask)
        output = self.decoder(output_embedded) # (batch_size, target.shape[1], self.d_model)
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

    def __init__(self, d_model, device, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model).to(device=device).float()
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)] for pos in range(max_len)])
        pe[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2])).to(device=device)
        pe[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2])).to(device=device)
        self.pe = pe.unsqueeze(0).detach()
        self.pe.requires_grad = False

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:, :x.size(1), :]
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
