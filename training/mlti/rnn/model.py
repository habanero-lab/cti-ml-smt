import torch
from torch.nn.utils.rnn import PackedSequence
import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum
import abc
from ..common import TypeDecoder


class MltiModel(nn.Module, abc.ABC):

    def __init__(self,
                 subname_vocab_size: int,
                 nonname_vocab_size: int,
                 emb_dim: int,
                 decoder_dim: int):
        super().__init__()
        self.subname_emb = nn.Embedding(subname_vocab_size, emb_dim)
        self.nonname_emb = nn.Embedding(nonname_vocab_size, emb_dim)
        self.type_decoder = TypeDecoder(decoder_dim)

    @abc.abstractmethod
    def encode(self,
               tokens: PackedSequence,
               names: torch.Tensor,
               subname_indices: torch.Tensor,
               infer_name_ids: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self,
                tokens: PackedSequence,
                names: torch.Tensor,
                subname_indices: torch.Tensor,
                infer_name_ids: torch.Tensor,
                type_seqs: list[torch.Tensor]):
        type_state = self.encode(
            tokens, names, subname_indices, infer_name_ids)
        return self.type_decoder(type_state, type_seqs)

    def infer(self,
              tokens: PackedSequence,
              names: torch.Tensor,
              subname_indices: torch.Tensor,
              infer_name_ids: torch.Tensor,
              beam_size: int,
              max_type_length: int):
        h = self.encode(tokens, names, subname_indices, infer_name_ids)
        return self.type_decoder.infer(h, beam_size, max_type_length)


class RNNLayer(nn.Module):

    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.rnn = nn.GRU(model_dim,
                          model_dim // 2,
                          num_layers=2,
                          bidirectional=True,
                          dropout=dropout)

    def forward(self,
                state: PackedSequence,
                is_name: torch.Tensor,
                is_nonname: torch.Tensor,
                name_ids: torch.Tensor,
                return_name_state: bool = True):
        state = PackedSequence(
            data=self.norm(state.data),
            batch_sizes=state.batch_sizes,
            sorted_indices=state.sorted_indices,
            unsorted_indices=state.unsorted_indices,
        )

        output, _ = self.rnn(state)
        data = output.data

        data = state.data + data

        name_state = scatter_mean(data[is_name], name_ids, dim=0)

        if return_name_state:
            return name_state
        else:
            x = torch.empty((is_name.size(0), name_state.size(1)),
                            dtype=torch.float,
                            device=is_name.device)
            x[is_name] = name_state[name_ids]
            x[is_nonname] = data[is_nonname]
            return PackedSequence(
                data=x,
                batch_sizes=state.batch_sizes,
                sorted_indices=state.sorted_indices,
                unsorted_indices=state.unsorted_indices,
            )


class RNNModel(MltiModel):

    def __init__(self, subname_vocab_size: int, nonname_vocab_size: int,
                 model_dim: int, num_layers: int, dropout: float):
        super().__init__(subname_vocab_size, nonname_vocab_size,
                         emb_dim=model_dim, decoder_dim=model_dim)

        self.layers = nn.ModuleList(
            [RNNLayer(model_dim, dropout) for _ in range(num_layers)])

        self.type_ffn = nn.Sequential(nn.LayerNorm(model_dim),
                                      nn.Linear(model_dim, model_dim))

    def encode(self,
               tokens: PackedSequence,
               names: torch.Tensor,
               subname_indices: torch.Tensor,
               infer_name_ids: torch.Tensor):
        name_emb = scatter_sum(self.subname_emb(names), subname_indices, dim=0)

        data = tokens.data
        is_name = data[:, 0] == 0
        is_nonname = ~is_name
        name_ids = data[is_name, 1]

        x = torch.empty((data.size(0), name_emb.size(1)),
                        dtype=torch.float,
                        device=name_emb.device)
        x[is_name] = name_emb[name_ids]
        x[is_nonname] = self.nonname_emb(data[is_nonname, 1])
        state = PackedSequence(
            data=x,
            batch_sizes=tokens.batch_sizes,
            sorted_indices=tokens.sorted_indices,
            unsorted_indices=tokens.unsorted_indices,
        )

        for i in range(len(self.layers)):
            return_symbol_state = i == len(self.layers) - 1
            state = self.layers[i](state, is_name, is_nonname,
                                   name_ids, return_symbol_state)

        name_state = state
        type_state = name_state[infer_name_ids]
        type_state = self.type_ffn(type_state)

        return type_state


class DeepTyperModel(MltiModel):

    def __init__(self, subname_vocab_size: int, nonname_vocab_size: int,
                 model_dim: int):
        emb_dim = 300
        rnn_dim = 650

        super().__init__(subname_vocab_size, nonname_vocab_size,
                         emb_dim=emb_dim, decoder_dim=model_dim)

        self.layer_norm = nn.LayerNorm(emb_dim)

        self.encoder = nn.GRU(emb_dim, rnn_dim // 2, bidirectional=True)
        self.recorder = nn.GRU(rnn_dim, rnn_dim // 2, bidirectional=True)

        self.type_ffn = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(rnn_dim, model_dim)
        )

    def encode(self,
               tokens: PackedSequence,
               names: torch.Tensor,
               subname_indices: torch.Tensor,
               infer_name_ids: torch.Tensor):
        name_emb = scatter_sum(self.subname_emb(names), subname_indices, dim=0)

        data = tokens.data
        is_name = data[:, 0] == 0
        is_nonname = ~is_name
        name_ids = data[is_name, 1]

        x = torch.empty((data.size(0), name_emb.size(1)),
                        dtype=torch.float,
                        device=name_emb.device)
        x[is_name] = name_emb[name_ids]
        x[is_nonname] = self.nonname_emb(data[is_nonname, 1])
        inp = PackedSequence(
            data=self.layer_norm(x),
            batch_sizes=tokens.batch_sizes,
            sorted_indices=tokens.sorted_indices,
            unsorted_indices=tokens.unsorted_indices,
        )

        enc, _ = self.encoder(inp)

        # Consistency layer
        enc_data = enc.data
        name_enc_mean = scatter_mean(enc_data[is_name], name_ids, dim=0)
        x = torch.empty_like(enc_data)
        x[is_name] = enc_data[is_name] + name_enc_mean[name_ids]
        x[is_nonname] = enc_data[is_nonname]
        enc = PackedSequence(
            data=x,
            batch_sizes=enc.batch_sizes,
            sorted_indices=enc.sorted_indices,
            unsorted_indices=enc.unsorted_indices,
        )

        rec, _ = self.recorder(enc)

        name_state = scatter_mean(rec.data[is_name], name_ids, dim=0)
        type_state = name_state[infer_name_ids]
        type_state = self.type_ffn(type_state)

        return type_state
