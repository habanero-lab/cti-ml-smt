import re
import argparse
from pathlib import Path
import ast
import pickle


type_names = ["Array", "List", "Dict", "bool", "int", "float"]
dtype_names = {"bool", "int", "float"}


class ParsingFailure(Exception):
    pass


def consume_token(s: str, t: str) -> str:
    try:
        i = 0
        while s[i] == " ":
            i += 1
        if s[i : i + len(t)] == t:
            i += len(t)
        else:
            raise ParsingFailure
        while i < len(s) and s[i] == " ":
            i += 1
    except IndexError:
        raise ParsingFailure
    return s[i:]


def parse_uint(s: str) -> tuple[int, str]:
    i = 0
    while s[i].isdigit():
        i += 1
    if i == 0:
        raise ParsingFailure
    u = int(s[:i])
    while s[i] == " ":
        i += 1
    return u, s[i:]


def parse_type(s: str) -> tuple[list[str], str]:
    match = re.search(f'^{"|".join(type_names)}', s)
    if match is None:
        raise ParsingFailure
    name = match.group()
    s = s[match.end() :]
    seq: list[str] = []
    if name == "Array":
        seq.append("array")
        s = consume_token(s, "(")
        t, s = parse_type(s)
        if t[0] not in dtype_names:
            raise ParsingFailure
        assert len(t) == 1
        seq += t
        s = consume_token(s, ",")
        ndim, s = parse_uint(s)
        seq.append(str(ndim))
        s = consume_token(s, ")")
        return seq, s
    elif name == "List":
        seq.append("list")
        s = consume_token(s, "(")
        t, s = parse_type(s)
        seq += t
        s = consume_token(s, ")")
        return seq, s
    elif name == "Dict":
        seq.append("dict")
        s = consume_token(s, "(")
        t, s = parse_type(s)
        if t[0] != "int":
            raise ParsingFailure
        s = consume_token(s, ",")
        t, s = parse_type(s)
        seq += t
        s = consume_token(s, ")")
        return seq, s
    else:
        if name not in dtype_names:
            raise ParsingFailure
        seq.append(name)
        return seq, s


def parse_response(response: str, src: str):
    tree: ast.Module = ast.parse(src)
    func_arg_names: list[list[str]] = []
    for fdef in tree.body:
        if not isinstance(fdef, ast.FunctionDef):
            continue
        arg_names = []
        for arg in fdef.args.args:
            arg_names.append(arg.arg)
        func_arg_names.append(arg_names)

    arg_name_type_strs: list[tuple[str, list[str]]] = []
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        arg_name, types_str = line.split(":")
        arg_name = arg_name.strip()
        type_strs = [s.strip() for s in types_str.split("|")]
        arg_name_type_strs.append((arg_name, type_strs))

    type_seqs: list[list[list[str]]] = []
    type_map = {}
    i = 0
    for arg_names in func_arg_names:
        for arg_name in arg_names:
            if i == len(arg_name_type_strs):
                if arg_name in type_map:
                    type_seqs.append(type_map[arg_name])
                    continue
                return None
            if arg_name != arg_name_type_strs[i][0]:
                if arg_name in type_map:
                    type_seqs.append(type_map[arg_name])
                    continue
                print(arg_name, arg_name_type_strs[i][0])
                raise ParsingFailure
            type_strs = arg_name_type_strs[i][1]
            i += 1
            ts = []
            for j in range(len(type_strs)):
                type_str = type_strs[j]
                try:
                    t, rem = parse_type(type_str)
                    assert not rem
                    ts.append(t)
                except ParsingFailure:
                    if j == len(type_strs) - 1 and i == len(arg_name_type_strs) - 1:
                        return None
                    raise ParsingFailure
            type_seqs.append(ts)
            type_map[arg_name] = ts

    return type_seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--responses", "-r", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--round", "-rd", type=int, required=True)

    args = parser.parse_args()

    input_path: Path = args.input
    if input_path.is_dir():
        paths = list(input_path.glob("**/*.pydd"))
    else:
        paths = [input_path]

    name_to_type_seqs = {}

    for path in paths:
        path = path.resolve()
        src = path.read_text()
        name = (path.parent.name, path.name)
        response_path: Path = (
            args.responses / name[0] / (name[1] + f".{args.round}.txt")
        )
        response = response_path.read_text()
        print(f"Parsing {name}")
        type_seqs = parse_response(response, src)
        assert type_seqs is not None
        for i, ts in enumerate(type_seqs):
            s = set()
            for t in ts:
                tt = tuple(t)
                if tt in s:
                    print(i)
                    print(ts, t)
                    assert False
                s.add(tt)
        name_to_type_seqs[name] = type_seqs

    with open(args.output, "wb") as f:
        pickle.dump(name_to_type_seqs, f)


if __name__ == "__main__":
    main()
