import torch
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from transformers import T5EncoderModel
from transformers import BatchEncoding
from ..common import TypeDecoder


class MltiModel(nn.Module):

    def __init__(self, encoder_name: str, decoder_dim: int):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(encoder_name)
        encoder_dim = self.encoder.config.hidden_size

        self.linear = nn.Linear(encoder_dim, decoder_dim)
        self.type_decoder = TypeDecoder(decoder_dim)

    def encode(self, input_encoding: BatchEncoding, type_token_ids: torch.Tensor):
        encoder_output: BaseModelOutput = self.encoder(**input_encoding)

        # (batch_size, length, encoder_dim)
        h = encoder_output.last_hidden_state
        # type_token_ids: (2, num_types)
        # (num_types, decoder_dim)
        type_state = self.linear(h[type_token_ids[0], type_token_ids[1]])

        return type_state

    def forward(self, input_encoding: BatchEncoding, type_token_ids: torch.Tensor,
                type_seqs: list[torch.Tensor]):
        type_state = self.encode(input_encoding, type_token_ids)
        return self.type_decoder(type_state, type_seqs)

    def infer(self, input_encoding: BatchEncoding, type_token_ids: torch.Tensor,
              beam_size: int, max_type_length: int):
        h = self.encode(input_encoding, type_token_ids)
        return self.type_decoder.infer(h, beam_size, max_type_length)
