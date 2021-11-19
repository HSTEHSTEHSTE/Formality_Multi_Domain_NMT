import torch
from fairseq.models.transformer.transformer_encoder import TransformerEncoder
from fairseq.models.transformer.transformer_decoder import TransformerDecoder
from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.data.dictionary import Dictionary

# Build config objects
config = TransformerConfig # default config

# Build dictionary
ja_dict = Dictionary()

# Build encoder and decoder objects
encoder = TransformerEncoder(config, ja_dict)
decoder = TransformerDecoder(config, ja_dict)


