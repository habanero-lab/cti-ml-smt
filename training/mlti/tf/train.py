import logging
import pickle
import functools
from pathlib import Path
import tqdm
import torch
from torch.utils.data import DataLoader
import transformers
import numpy as np
from .model import MltiModel
from .dataset import MltiDataset
from ..common import eval_beam


def train(dataset_path: Path, model_dir: Path, encoder_name: str, decoder_dim: int,
          batch_size: int, subbatch_size: int, infer_batch_size: int, lr: float, num_epochs: int,
          lr_scheduler_name: str, use_gpu: bool, beam_size: int, max_type_length: int,
          seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

    assert batch_size % subbatch_size == 0
    accum_steps = batch_size // subbatch_size

    model_dir.mkdir(exist_ok=True)

    with open(dataset_path, 'rb') as f:
        train_data, val_data, test_data = pickle.load(f)

    train_dataset = MltiDataset(train_data)
    val_dataset = MltiDataset(val_data)

    tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_name)
    collate = functools.partial(MltiDataset.collate, tokenizer=tokenizer)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=subbatch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  collate_fn=collate)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=infer_batch_size,
                                shuffle=False,
                                pin_memory=True,
                                collate_fn=collate)

    device = torch.device('cuda') \
        if use_gpu and torch.cuda.is_available() else torch.device('cpu')

    model = MltiModel(encoder_name, decoder_dim)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if lr_scheduler_name == 'constant':
        lr_scheduler = transformers.get_constant_schedule(optimizer=optimizer)
    elif lr_scheduler_name == 'linear':
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_data) // accum_steps) * num_epochs
        )
    elif lr_scheduler_name == 'linear_with_warmup':
        num_iters = (len(train_data) // accum_steps) * num_epochs
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(num_iters * 0.1),
            num_training_steps=num_iters
        )
    else:
        assert False

    best_loss = float('inf')
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        _run_epoch(epoch,
                   model,
                   train_dataloader,
                   accum_steps,
                   optimizer,
                   lr_scheduler,
                   beam_size,
                   max_type_length,
                   device,
                   training=True)

        torch.save(model, str(model_dir / f'model.ep{epoch}.pt'))

        with torch.inference_mode():
            val_loss = \
                _run_epoch(epoch, model, val_dataloader, accum_steps,
                           optimizer, lr_scheduler,
                           beam_size, max_type_length,
                           device, training=False)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model, str(model_dir / f'model.pt'))

        logging.info(f' * Best epoch: {best_epoch}, loss: {best_loss:.6f}\n')


def test(dataset_path: Path, model_path: Path, encoder_name: str,
         batch_size: int, use_gpu: bool, beam_size: int, max_type_length: int):
    device = torch.device('cuda') \
        if use_gpu and torch.cuda.is_available() else torch.device('cpu')

    model: MltiModel = torch.load(model_path, map_location=device)
    model.eval()

    with open(dataset_path, 'rb') as f:
        _, _, test_data = pickle.load(f)

    test_dataset = MltiDataset(test_data)

    tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_name)
    collate = functools.partial(MltiDataset.collate, tokenizer=tokenizer)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 collate_fn=collate)

    correct = np.zeros(beam_size, dtype=np.int32)
    basic_correct = np.zeros(beam_size, dtype=np.int32)
    nested_correct = np.zeros(beam_size, dtype=np.int32)
    total = 0
    basic_total = 0
    nested_total = 0

    with tqdm.tqdm(test_dataloader) as pbar:
        for input_encoding, token_ids, type_seqs in pbar:
            input_encoding = input_encoding.to(device)
            token_ids = token_ids.to(device)

            batch_preds = model.infer(
                input_encoding, token_ids, beam_size, max_type_length)
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


def _run_epoch(epoch: int, model: MltiModel, dataloader: DataLoader, accum_steps: int,
               optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
               beam_size: int, max_type_length: int,
               device: torch.device, training: bool):
    model.train(training)

    total_loss = 0

    if training:
        optimizer.zero_grad()
    else:
        correct = np.zeros(beam_size, dtype=np.int32)
        basic_correct = np.zeros(beam_size, dtype=np.int32)
        nested_correct = np.zeros(beam_size, dtype=np.int32)
        total = 0
        basic_total = 0
        nested_total = 0

    with tqdm.tqdm(dataloader) as pbar:
        for i, (input_encoding, token_ids, type_seqs) in enumerate(pbar):
            input_encoding = input_encoding.to(device)
            token_ids = token_ids.to(device)
            type_seqs_cpu = type_seqs
            type_seqs = [x.to(device) for x in type_seqs]

            logps = model(input_encoding, token_ids, type_seqs)
            loss = -torch.mean(logps)

            total_loss += loss.item()

            pbar.set_description(f'Epoch {epoch}, loss: {loss.item():.6f}')

            if training:
                loss = loss / accum_steps
                loss.backward()
                if (i + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            else:
                batch_preds = model.infer(
                    input_encoding, token_ids, beam_size, max_type_length)
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
