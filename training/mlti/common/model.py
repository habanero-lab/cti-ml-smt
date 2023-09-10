import torch
from torch.nn.utils.rnn import pack_sequence, PackedSequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pydd_types import type_vocab, type_tokens


class TypeDecoder(nn.Module):

    def __init__(self, model_dim: int):
        super().__init__()

        type_vocab_size = len(type_vocab)
        self.type_emb = nn.Embedding(type_vocab_size + 2, model_dim)
        self.type_eos_id = type_vocab_size
        self.type_sos_id = type_vocab_size + 1

        self.type_rnn = nn.GRU(model_dim, model_dim)
        self.proj = nn.Linear(model_dim, type_vocab_size + 1)

    def forward(self, h: torch.Tensor, type_seqs: list[torch.Tensor]):
        input_type_seqs = pack_sequence(
            [F.pad(seq, (1, 0), value=self.type_sos_id) for seq in type_seqs],
            enforce_sorted=False)
        output_type_seqs = pack_sequence(
            [F.pad(seq, (0, 1), value=self.type_eos_id) for seq in type_seqs],
            enforce_sorted=False)

        input_type_seqs = PackedSequence(
            data=self.type_emb(input_type_seqs.data),
            batch_sizes=input_type_seqs.batch_sizes,
            sorted_indices=input_type_seqs.sorted_indices,
            unsorted_indices=input_type_seqs.unsorted_indices)

        out, _ = self.type_rnn(input_type_seqs, h.unsqueeze(0))
        logp = F.log_softmax(self.proj(out.data), dim=-1)

        logps = torch.gather(
            logp, dim=1, index=output_type_seqs.data.unsqueeze(1)).squeeze(1)

        return logps

    def infer(self, h: torch.Tensor, beam_size: int, max_type_length: int) \
            -> list[list[tuple[float, list[int]]]]:
        preds = [[(0, False, h[i], [self.type_sos_id])]
                 for i in range(h.size(0))]

        for _ in range(max_type_length):
            x = []
            hs = []
            for i in range(len(preds)):
                for _, done, h_prev, pred_prev in preds[i]:
                    if not done:
                        x.append(pred_prev[-1])
                        hs.append(h_prev)
            if not x:
                break

            x = torch.tensor(x, dtype=torch.long, device=h.device)
            h = torch.stack(hs).unsqueeze(0)
            out, h = self.type_rnn(self.type_emb(x).unsqueeze(0), h)
            h = h.squeeze(0)
            logits = self.proj(out.squeeze(0))
            logps = F.log_softmax(logits, dim=-1)
            logps = logps.detach().cpu().numpy()

            k = 0
            for i in range(len(preds)):
                pred_new = []
                for neg_logp_prev, done, h_prev, pred_prev in preds[i]:
                    if done:
                        pred_new.append(
                            (neg_logp_prev, done, h_prev, pred_prev))
                        continue
                    for j, logp in enumerate(logps[k]):
                        if j == self.type_eos_id:
                            done = True
                        pred_new.append((neg_logp_prev - logp, done, h[k],
                                         pred_prev + [j]))
                    k += 1
                pred_new = [x for x in pred_new if self.is_legal_prefix(x[-1])]
                pred_new.sort(key=lambda x: x[0])
                preds[i] = pred_new[:beam_size]
            assert k == logps.shape[0]

        ret = [[] for _ in range(len(preds))]
        for i in range(len(preds)):
            for neg_logp, done, _, pred in preds[i]:
                if done:
                    ret[i].append((-neg_logp, pred[1:-1]))

        return ret

    def is_legal_prefix(self, prefix: list[int]):
        x = prefix[1:]
        if x[-1] == self.type_eos_id:
            x = x[:-1]
            x = [type_tokens[t] for t in x]
            return is_legal_type_seq(x)
        else:
            x = [type_tokens[t] for t in x]
            return is_legal_type_seq_prefix(x)


def is_legal_type_seq_prefix(seq):
    if not seq:
        return True
    name = seq[0]
    if name == 'list':
        return is_legal_type_seq_prefix(seq[1:])
    elif name == 'dict':
        return is_legal_type_seq_prefix(seq[1:])
    elif name == 'array':
        if len(seq) > 3:
            return False
        if len(seq) > 1:
            if seq[1] not in ('int', 'float', 'bool'):
                return False
            if len(seq) > 2:
                if not seq[2].isdigit():
                    return False
        return True
    elif name in ('int', 'float', 'bool'):
        return len(seq) == 1
    return False


def is_legal_type_seq(seq):
    if not seq:
        return False
    name = seq[0]
    if name == 'list':
        return is_legal_type_seq(seq[1:])
    elif name == 'dict':
        return is_legal_type_seq(seq[1:])
    elif name == 'array':
        if len(seq) != 3:
            return False
        if seq[1] not in ('int', 'float', 'bool'):
            return False
        if not seq[2].isdigit():
            return False
        return True
    elif name in ('int', 'float', 'bool'):
        return len(seq) == 1
    return False
