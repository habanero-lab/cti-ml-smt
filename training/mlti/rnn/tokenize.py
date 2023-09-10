from typing import NamedTuple
import enum
import ast
import token
import keyword
import autopep8
import strip_hints
import asttokens
import asttokens.util


class TokenKind(enum.Enum):
    Op = 'O'
    Num = 'N'
    Str = 'S'
    Name = 'I'
    Keyword = 'K'
    Start = '<'
    End = '>'
    Other = 'T'


class Token(NamedTuple):
    kind: TokenKind
    string: str


_sos_token = Token(TokenKind.Start, '')
_eos_token = Token(TokenKind.End, '')


def tokenize(src: str):
    src = autopep8.fix_code(strip_type_hints(strip_comments(src)))
    ast_tokens = asttokens.ASTTokens(src)

    tokens: list[Token] = [_sos_token]

    for tok in ast_tokens.tokens:
        token_type = tok.type
        token_str = tok.string
        if token_type == 0 or token_type > token.ASYNC:
            continue
        if token_type == token.OP:
            token_kind = TokenKind.Op
        elif token_type == token.NUMBER:
            token_kind = TokenKind.Num
            token_str = type(ast.literal_eval(token_str)).__name__.upper()
        elif token_type == token.STRING:
            token_kind = TokenKind.Str
            token_str = 'STR'
        elif token_type == token.NAME:
            if keyword.iskeyword(token_str):
                token_kind = TokenKind.Keyword
            else:
                token_kind = TokenKind.Name
        else:
            token_kind = TokenKind.Other
            token_str = token.tok_name[token_type]

        tokens.append(Token(token_kind, token_str))

    tokens.append(_eos_token)

    return tokens


def strip_type_hints(src: str):
    return strip_hints.strip_string_to_string(src, to_empty=True, strip_nl=True)


def strip_comments(src: str):
    ast_tokens = asttokens.ASTTokens(src, parse=True)
    replacements = []
    for t in ast_tokens.tokens:
        if t.type == token.COMMENT or t.type == token.TYPE_COMMENT:
            replacements.append((t.startpos, t.endpos, ''))
    for node in asttokens.util.walk(ast_tokens.tree):
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Str):
                startpos, endpos = ast_tokens.get_text_range(node)
                replacements.append((startpos, endpos, ''))
    return asttokens.util.replace(src, replacements)
