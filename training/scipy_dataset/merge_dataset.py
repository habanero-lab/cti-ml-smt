import argparse
import io
from pathlib import Path
import pickle
import tqdm
import multiprocessing
import tempfile
import itertools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', type=Path, required=True)
    parser.add_argument('--output', '-o', type=Path, required=True)
    parser.add_argument('--keep-bitwidth', '-kb', action='store_true')
    parser.add_argument('--num-processes', '-np', type=int, default=None)
    args = parser.parse_args()

    data_files = list(args.input_dir.glob('**/*.pkl'))

    tmp_paths = []
    with multiprocessing.Pool(args.num_processes) as pool:
        for tmp_path in tqdm.tqdm(pool.imap_unordered(deduplicate,
                                                      zip(data_files, itertools.repeat(args))),
                                  total=len(data_files)):
            tmp_paths.append(tmp_path)

    hashes = set()
    with open(args.output, 'wb') as out:
        for tmp_path in tqdm.tqdm(tmp_paths):
            process_file(tmp_path, hashes, out, args)


def deduplicate(arguments) -> Path:
    data_file, args = arguments
    hashes = set()
    out = tempfile.NamedTemporaryFile('wb', delete=False)
    out_path = Path(out.name).resolve()
    out.close()
    with open(out_path, 'wb') as out:
        process_file(data_file, hashes, out, args)
    return out_path


def process_file(path: Path, hashes: set[int], out: io.BufferedWriter, args):
    with open(path, 'rb') as f:
        while True:
            try:
                src, arg_types = pickle.load(f)[:2]
            except EOFError:
                break
            except Exception as e:
                print(repr(e))
                break

            src = outdent(src)
            arg_types = {k: v for k, v in arg_types.items() if v is not None}

            if not args.keep_bitwidth:
                arg_types = strip_bitwidth(arg_types)

            h = hash((src, frozenset(arg_types.items())))
            if h in hashes:
                continue
            hashes.add(h)
            pickle.dump((src, arg_types), out)


bitwidth_map = {
    'int32': 'int',
    'int64': 'int',
    'float32': 'float',
    'float64': 'float',
    'bool8': 'bool',
    'bool_': 'bool',
}


def strip_bitwidth(arg_types: dict[str, tuple]):
    ret = {}
    for k, v in arg_types.items():
        ret[k] = tuple([bitwidth_map.get(x, x) for x in v])
    return ret


def outdent(src: str):
    lines = src.splitlines()
    indent = len(src)
    new_lines = []
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        i = 0
        n = len(line)
        while i < n:
            if line[i] == ' ' or line[i] == '\t':
                i += 1
                continue
            if line[i] == '#':
                i = -1
            break
        if i < 0:
            continue
        indent = min(indent, i)
        new_lines.append(line)
    return '\n'.join(line[indent:] for line in new_lines)


if __name__ == '__main__':
    main()
