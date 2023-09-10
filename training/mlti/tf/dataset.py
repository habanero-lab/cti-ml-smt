import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase


DataItemType = tuple[BatchEncoding, np.ndarray, list[np.ndarray]]


class MltiDataset(Dataset):

    def __init__(self, data: list[DataItemType]):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i: int):
        input_encoding, token_ids, type_seqs = self._data[i]

        input_encoding = input_encoding.convert_to_tensors('pt')
        token_ids = torch.from_numpy(token_ids)
        type_seqs = [torch.from_numpy(seq) for seq in type_seqs]

        return input_encoding, token_ids, type_seqs

    @staticmethod
    def collate(batch, tokenizer: PreTrainedTokenizerBase):
        input_encoding, token_ids, type_seqs = zip(*batch)

        input_encoding = tokenizer.pad(input_encoding)
        locations = []
        for i, t in enumerate(token_ids):
            locations.append(
                torch.vstack((torch.full_like(t, fill_value=i), t)))
        token_ids = torch.cat(locations, dim=1)
        type_seqs = [seq for l in type_seqs for seq in l]

        return input_encoding, token_ids, type_seqs
