from typing import Iterable
import regex


def _split_by_underscore(words: Iterable[str]):
    for word in words:
        for subword in word.split('_'):
            if subword:
                yield subword


_camel_pattern = regex.compile(
    r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[0-9])(?=[A-Za-z])',
    flags=regex.V1)


def _split_by_camel_case(words: Iterable[str]):
    for word in words:
        for subword in _camel_pattern.splititer(word):
            if subword:
                yield subword


def _lowercase(words: Iterable[str]):
    for word in words:
        yield word.lower()


def split_name(name: str):
    subnames = list(
        _lowercase(_split_by_camel_case(_split_by_underscore([name]))))
    if not subnames:
        subnames = ['_']
    return subnames
