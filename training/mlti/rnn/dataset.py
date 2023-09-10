import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence


DataItemType = tuple[np.ndarray, list[np.ndarray],
                     np.ndarray, list[np.ndarray]]


class MltiDataset(Dataset):

    def __init__(self, data: list[DataItemType]):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i: int):
        tokens_arr, name_arrs, infer_name_ids, type_seqs_arr = self._data[i]

        tokens = torch.from_numpy(tokens_arr)
        names = [torch.from_numpy(name_arr) for name_arr in name_arrs]
        infer_name_ids = torch.from_numpy(infer_name_ids)
        type_seqs = [torch.from_numpy(seq) for seq in type_seqs_arr]

        return tokens, names, infer_name_ids, type_seqs

    @staticmethod
    def collate(batch):
        batch_names = []
        batch_subname_indices = []
        batch_tokens = []
        batch_infer_name_ids = []
        batch_type_seqs = []

        for tokens, names, infer_name_ids, type_seqs in batch:
            prev_num_names = len(batch_names)
            for i, name in enumerate(names):
                subname_indices = torch.full_like(name, prev_num_names + i)
                batch_subname_indices.append(subname_indices)
            batch_names += names

            tokens_tensor = tokens.clone()
            is_name = tokens_tensor[:, 0] == 0
            tokens_tensor[is_name, 1] += prev_num_names
            batch_tokens.append(tokens_tensor)

            batch_infer_name_ids.append(infer_name_ids + prev_num_names)

            batch_type_seqs += type_seqs

        return pack_sequence(batch_tokens, enforce_sorted=False), \
            torch.cat(batch_names), torch.cat(batch_subname_indices), \
            torch.cat(batch_infer_name_ids), batch_type_seqs
