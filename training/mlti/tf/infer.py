import numpy as np
from pathlib import Path
import torch
import transformers
from .dataset import MltiDataset
from .model import MltiModel
from .preprocess import preprocess_pydd_src
from ..common import eval_pydd


def infer(paths: list[Path], model_path: Path, encoder_name: str,
          top_k: int, beam_size: int, max_type_length: int):
    tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_name)

    model: MltiModel = torch.load(model_path, map_location='cpu')
    model.eval()

    data = []
    info = []

    for path in paths:
        batch, func_param_to_infer_id, func_names = \
            preprocess_pydd_src(path.read_text(), tokenizer)

        func_type_seqs = []
        for batch_encoding, token_ids, type_seqs_arr in batch:
            batch_encoding = batch_encoding.convert_to_tensors('pt')
            token_ids = torch.from_numpy(token_ids)
            data.append((batch_encoding, token_ids, []))
            func_type_seqs.append(type_seqs_arr)

        info.append((path, func_names, func_param_to_infer_id, func_type_seqs))

    input_encoding, token_ids, _ = MltiDataset.collate(data, tokenizer)

    with torch.inference_mode():
        preds = model.infer(input_encoding, token_ids,
                            beam_size, max_type_length)

    eval_pydd(info, preds, top_k)
