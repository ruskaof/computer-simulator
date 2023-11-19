from __future__ import annotations

import dataclasses
import sys
from enum import Enum, auto
from pathlib import Path
from typing import cast

from computer_simulator.translator import ProgramChar, Token, NoValueToken, ValueToken, TranslatorException
from computer_simulator.translator.tokenizer import tokenize


class Opcode(Enum):
    pass


@dataclasses.dataclass
class Operation:
    def __init__(self, opcode: Opcode, args: list):
        self.opcode: Opcode = opcode
        self.args: list = args


def parse_expression(tokens: list[Token], idx: int) -> int:
    pass


def handle_no_value_token(token: NoValueToken, tokens: list[Token], idx: int) -> int:


def handle_value_token(token: ValueToken, tokens: list[Token], idx: int) -> int:
    pass


def run_translator(tokens: list[Token]) -> str:
    idx: int = 0

    while idx < len(tokens):
        token: Token = tokens[idx]
        if isinstance(token, NoValueToken) and cast(NoValueToken, token).token_type == NoValueToken.Type.OPEN_BRACKET:
            idx = parse_expression(tokens, idx + 1)
            if (idx >= len(tokens) or not isinstance(tokens[idx], NoValueToken) or
                    cast(NoValueToken, tokens[idx]).token_type != NoValueToken.Type.CLOSE_BRACKET):
                raise TranslatorException(tokens[idx].starting_char, ")")
            idx += 1
        else:
            raise TranslatorException(tokens[idx].starting_char, "expression")


def read_program(source: str) -> list[ProgramChar]:
    with open(source, encoding="utf-8") as f:
        code: str = f.read()

    lines: list[str] = code.split("\n")
    chars: list[ProgramChar] = []

    for line_idx, line in enumerate(lines):
        for char_idx, char in enumerate(line):
            chars.append(ProgramChar(char, line_idx, char_idx))

    return chars


def main(source: str, target: str) -> None:
    with open(target, "w", encoding="utf-8") as f:
        f.write(run_translator(tokenize(read_program(source))))


if __name__ == "__main__":
    assert len(sys.argv) == 3, f"Usage: python3 {Path(__file__).name} <source> <target>"
    main(sys.argv[1], sys.argv[2])
