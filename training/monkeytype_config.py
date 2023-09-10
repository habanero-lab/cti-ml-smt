import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Optional
import inspect
import os
import logging
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'MonkeyType'))
if TYPE_CHECKING:
    from MonkeyType.monkeytype.config import DefaultConfig
    from MonkeyType.monkeytype.db.base import CallTraceStore
    from MonkeyType.monkeytype.tracing import CallTrace
    from MonkeyType.monkeytype.compat import is_generic, name_of_generic
else:
    from monkeytype.config import DefaultConfig
    from monkeytype.db.base import CallTraceStore
    from monkeytype.tracing import CallTrace
    from monkeytype.compat import is_generic, name_of_generic


def is_list(typ: type) -> bool:
    return is_generic(typ) and name_of_generic(typ) == "List"


def is_dict(typ: type) -> bool:
    return is_generic(typ) and name_of_generic(typ) == "Dict"


primitive_type_encoding: Dict[type, tuple[str]] = {
    int: ('int', ),
    np.int32: ('int32', ),
    np.int64: ('int64', ),
    float: ('float', ),
    np.float32: ('float32', ),
    np.float64: ('float64', ),
    bool: ('bool', ),
    np.bool8: ('bool', ),
}


def encode_type(typ: Optional[type]) -> Optional[tuple]:
    if typ is None:
        return None
    enc = primitive_type_encoding.get(typ, None)
    if enc is not None:
        return enc
    if is_list(typ):
        elem_type_enc = encode_type(getattr(typ, "__args__")[0])
        if elem_type_enc is None:
            return None
        return ('list', elem_type_enc)
    if is_dict(typ):
        args = getattr(typ, "__args__")
        # if args[0] != int:
        #     return None
        elem_type_enc = encode_type(args[1])
        if elem_type_enc is None:
            return None
        return ('dict', elem_type_enc)
    type_name = getattr(typ, '__name__', None)
    if type_name is None:
        return None
    if type_name.startswith('Array,'):
        try:
            _, dtype, rank = type_name.split(',')
        except:
            print(type_name)
            raise
        return ('array', dtype, int(rank))
    return None


class DatasetStore(CallTraceStore):

    def __init__(self, dataset_dir: str) -> None:
        self.dataset_dir_path = Path(dataset_dir).resolve()
        self.dataset_dir_path.mkdir(exist_ok=True)

    @classmethod
    def make_store(cls, dataset_path: str) -> "CallTraceStore":
        return cls(dataset_path)

    def add(self, traces: Iterable[CallTrace]) -> None:
        dataset_path = self.dataset_dir_path / f'{os.getpid()}.pkl'
        with open(dataset_path, 'ab') as f:
            for trace in traces:
                try:
                    src = inspect.getsource(trace.func)
                except OSError as e:
                    logging.warning('%s: %s', e, e.filename)
                    continue

                if len(src.splitlines()) < 5:
                    continue

                keep = False
                arg_types = {}
                for k, v in trace.arg_types.items():
                    enc = encode_type(v)
                    if enc is not None:
                        keep = True
                    arg_types[k] = enc
                # return_type = encode_type(trace.return_type)
                # if return_type is not None:
                #     keep = True
                if not keep:
                    continue

                # pickle.dump((src, arg_types, return_type), f)
                pickle.dump((src, arg_types), f)

    def filter(self, *args, **kwargs):
        return []

    def list_modules(self) -> list[str]:
        return []


class DatasetConfig(DefaultConfig):

    def trace_store(self) -> CallTraceStore:
        path = os.environ.get('DATASET_OUTPUT_DIR', 'dataset_raw')
        return DatasetStore.make_store(path)


CONFIG = DatasetConfig()
