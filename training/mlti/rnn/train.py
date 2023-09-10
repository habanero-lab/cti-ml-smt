import logging
import pickle
from pathlib import Path
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import MltiModel, RNNModel, DeepTyperModel
from .dataset import MltiDataset
from .preprocess import Vocab
from ..common import eval_beam
from ..common.pydd_types import type_vocab


def train(vocab_path: Path, dataset_path: Path, model_dir: Path,
          model_dim: int, num_layers: int, dropout: float,
          batch_size: int, lr: float, num_epochs: int, use_gpu: bool,
          beam_size: int, max_type_length: int, deep_typer: bool,
          seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_dir.mkdir(exist_ok=True)

    with open(vocab_path, 'rb') as f:
        vocab: Vocab = pickle.load(f)

    with open(dataset_path, 'rb') as f:
        train_data, val_data, test_data = pickle.load(f)

    train_dataset = MltiDataset(train_data)
    val_dataset = MltiDataset(val_data)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  collate_fn=MltiDataset.collate)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                collate_fn=MltiDataset.collate)

    device = torch.device('cuda') \
        if use_gpu and torch.cuda.is_available() else torch.device('cpu')

    model: MltiModel
    if deep_typer:
        model = DeepTyperModel(len(vocab.subname_bpe.vocab), len(vocab.nonname_vocab),
                               model_dim)
    else:
        model = RNNModel(len(vocab.subname_bpe.vocab), len(vocab.nonname_vocab),
                         model_dim, num_layers, dropout)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        _run_epoch(epoch,
                   model,
                   train_dataloader,
                   optimizer,
                   beam_size,
                   max_type_length,
                   device,
                   training=True)

        torch.save(model, str(model_dir / f'model.ep{epoch}.pt'))

        with torch.inference_mode():
            val_loss = \
                _run_epoch(epoch, model, val_dataloader, optimizer,
                           beam_size, max_type_length,
                           device, training=False)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model, str(model_dir / f'model.pt'))

        logging.info(f' * Best epoch: {best_epoch}, loss: {best_loss:.6f}\n')


def _run_epoch(epoch: int, model: MltiModel, dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               beam_size: int, max_type_length: int,
               device: torch.device, training: bool):
    model.train(training)

    total_loss = 0

    if training:
        pass
    else:
        correct = np.zeros(beam_size, dtype=np.int32)
        basic_correct = np.zeros(beam_size, dtype=np.int32)
        nested_correct = np.zeros(beam_size, dtype=np.int32)
        total = 0
        basic_total = 0
        nested_total = 0

    with tqdm.tqdm(dataloader) as pbar:
        for tokens, names, subname_indices, infer_name_ids, type_seqs in pbar:
            tokens = tokens.to(device)
            names = names.to(device)
            subname_indices = subname_indices.to(device)
            infer_name_ids = infer_name_ids.to(device)
            type_seqs_cpu = type_seqs
            type_seqs = [x.to(device) for x in type_seqs]

            logps = model(tokens, names, subname_indices,
                          infer_name_ids, type_seqs)
            loss = -torch.mean(logps)

            total_loss += loss.item()

            pbar.set_description(f'Epoch {epoch}, loss: {loss.item():.6f}')

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                batch_preds = model.infer(
                    tokens, names, subname_indices, infer_name_ids, beam_size, max_type_length)
                r = eval_beam(batch_preds, type_seqs_cpu, beam_size)
                batch_correct, batch_total, \
                    batch_basic_correct, batch_basic_total, \
                    batch_nested_correct, batch_nested_total \
                    = r
                correct += batch_correct
                basic_correct += batch_basic_correct
                nested_correct += batch_nested_correct
                total += batch_total
                basic_total += batch_basic_total
                nested_total += batch_nested_total

    avg_loss = total_loss / len(dataloader)

    if training:
        logging.info(f'Epoch {epoch}\n'
                     f'- training avg loss: {avg_loss:.6f}')
    else:
        logging.info(f'Epoch {epoch}\n'
                     f'- validation avg loss: {avg_loss:.6f}\n'
                     f'- validation acc: {correct / total}\n'
                     f'- validation basic acc: {basic_correct / basic_total}\n'
                     f'- validation nested acc: {nested_correct / nested_total}\n')

    return avg_loss


def test(dataset_path: Path, model_path: Path,
         batch_size: int, use_gpu: bool, beam_size: int, max_type_length: int):
    device = torch.device('cuda') \
        if use_gpu and torch.cuda.is_available() else torch.device('cpu')

    model: MltiModel = torch.load(model_path, map_location=device)
    model.eval()

    with open(dataset_path, 'rb') as f:
        _, _, test_data = pickle.load(f)

    test_dataset = MltiDataset(test_data)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 collate_fn=MltiDataset.collate)

    correct = np.zeros(beam_size, dtype=np.int32)
    basic_correct = np.zeros(beam_size, dtype=np.int32)
    nested_correct = np.zeros(beam_size, dtype=np.int32)
    total = 0
    basic_total = 0
    nested_total = 0

    with tqdm.tqdm(test_dataloader) as pbar:
        for tokens, names, subname_indices, infer_name_ids, type_seqs in pbar:
            tokens = tokens.to(device)
            names = names.to(device)
            subname_indices = subname_indices.to(device)
            infer_name_ids = infer_name_ids.to(device)

            batch_preds = model.infer(
                tokens, names, subname_indices, infer_name_ids, beam_size, max_type_length)
            r = eval_beam(batch_preds, type_seqs, beam_size)
            batch_correct, batch_total, \
                batch_basic_correct, batch_basic_total, \
                batch_nested_correct, batch_nested_total \
                = r
            correct += batch_correct
            basic_correct += batch_basic_correct
            nested_correct += batch_nested_correct
            total += batch_total
            basic_total += batch_basic_total
            nested_total += batch_nested_total

    logging.info(f'acc: {correct / total}\n'
                 f'basic acc: {basic_correct / basic_total}\n'
                 f'nested acc: {nested_correct / nested_total}\n')


def test_freq(dataset_path: Path, vocab_path: Path, beam_size: int):
    with open(vocab_path, 'rb') as f:
        _ = pickle.load(f)
        freq_type_seqs: list[tuple[str]] = pickle.load(f)[:beam_size]

    preds = [[(0., [type_vocab[t] for t in type_seq])
              for type_seq in freq_type_seqs]]

    with open(dataset_path, 'rb') as f:
        _, _, test_data = pickle.load(f)

    test_dataset = MltiDataset(test_data)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 collate_fn=MltiDataset.collate)

    correct = np.zeros(beam_size, dtype=np.int32)
    basic_correct = np.zeros(beam_size, dtype=np.int32)
    nested_correct = np.zeros(beam_size, dtype=np.int32)
    total = 0
    basic_total = 0
    nested_total = 0

    with tqdm.tqdm(test_dataloader) as pbar:
        for tokens, names, subname_indices, infer_name_ids, type_seqs in pbar:
            batch_preds = preds * len(type_seqs)
            r = eval_beam(batch_preds, type_seqs, beam_size)
            batch_correct, batch_total, \
                batch_basic_correct, batch_basic_total, \
                batch_nested_correct, batch_nested_total \
                = r
            correct += batch_correct
            basic_correct += batch_basic_correct
            nested_correct += batch_nested_correct
            total += batch_total
            basic_total += batch_basic_total
            nested_total += batch_nested_total

    logging.info(f'acc: {correct / total}\n'
                 f'basic acc: {basic_correct / basic_total}\n'
                 f'nested acc: {nested_correct / nested_total}\n')
