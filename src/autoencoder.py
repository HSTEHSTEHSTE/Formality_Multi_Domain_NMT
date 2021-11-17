import torch
from fairseq.models.transformer import transformer_encoder
from fairseq.models.transformer import transformer_decoder
from fairseq.models.transformer import transformer_config

# Build config objects
config = transformer_config.TransformerConfig # default config

# Build encoder and decoder objects
encoder = transformer_encoder.TransformerEncoder(config)
decoder = transformer_decoder.TransformerDecoder(config)


