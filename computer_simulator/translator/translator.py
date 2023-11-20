from __future__ import annotations

import enum
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import cast

from computer_simulator.translator import ProgramChar, Token, NoValueToken, ValueToken, TranslatorException, \
    MemoryAddress
from computer_simulator.translator.tokenizer import tokenize


class Opcode(Enum):
    # M - memory address or direct value
    # AC - accumulator
    LD: int = auto()  # M -> AC


class Heap:
    _memory: list[int] = {}
    _variables: dict[str, MemoryAddress] = {}

    def allocate_string(self, value: str) -> MemoryAddress:
        self._memory.append(len(value) + 1)
        for char in value:
            self._memory.append(ord(char))

        return MemoryAddress(len(self._memory) - 1)

    def init_variable(self, name: str) -> MemoryAddress:
        self._variables[name] = MemoryAddress(len(self._memory))
        self._memory.append(0)
        return self._variables[name]

    def get_variable(self, name: str) -> MemoryAddress:
        return self._variables[name]


class Program:
    heap: Heap = Heap()
    operations: list[Operation] = []


@dataclass
class Operation:
    opcode: Opcode
    args: list[IntValue | StringValue]


@dataclass
class IntValue:
    value: int


@dataclass
class StringValue:
    memory_address: MemoryAddress


def handle_if(tokens: list[Token], idx: int, result: Program) -> int:
    idx += 1
    if idx >= len(tokens):
        raise RuntimeError("Unexpected end of program")

    parse_s_expression(tokens, idx, result)


def parse_expression_inner(tokens: list[Token], idx: int, result: Program) -> int:
    if idx >= len(tokens):
        return idx

    if isinstance(tokens[idx], NoValueToken):
        token = cast(NoValueToken, tokens[idx])
        if token.token_type == NoValueToken.Type.IF:
            return handle_if(tokens, idx, result)
        elif token.token_type == NoValueToken.Type.OPEN_BRACKET:
            return parse_expression_inner(tokens, idx + 1, result)
        elif token.token_type == NoValueToken.Type.CLOSE_BRACKET:
            return parse_expression(tokens, idx, result)
        else:
            raise RuntimeError("Unknown no value token type")
    else:
        return handle_atom(tokens, idx, result)

def parse_expression(tokens: list[Token], idx: int, result: Program) -> int:
    if idx >= len(tokens):
        return idx

    if isinstance(tokens[idx], NoValueToken):
        token = cast(NoValueToken, tokens[idx])
        if token.token_type == NoValueToken.Type.OPEN_BRACKET:
            return parse_expression_inner(tokens, idx + 1, result)



def handle_atom(tokens: list[Token], idx: int, result: Program) -> int:
    if idx >= len(tokens):
        return idx

    if isinstance(tokens[idx], ValueToken):
        token = cast(ValueToken, tokens[idx])
        if token.token_type == ValueToken.Type.INT:
            result.operations.append(Operation(Opcode.LD, [IntValue(int(token.value))]))
            return idx + 1
        elif token.token_type == ValueToken.Type.STRING:
            result.operations.append(Operation(Opcode.LD, [StringValue(result.heap.allocate_string(token.value))]))
            return idx + 1
        elif token.token_type == ValueToken.Type.IDENTIFIER:
            result.operations.append(Operation(Opcode.LD, [StringValue(result.heap.get_variable(token.value))]))
            return idx + 1
        else:
            raise RuntimeError("Unknown value token type")
    else:
        raise RuntimeError("Unknown token type")



def parse_s_expression(tokens: list[Token], idx: int, result: Program) -> int:
    if idx >= len(tokens):
        return idx

    if isinstance(tokens[idx],
                  NoValueToken and cast(NoValueToken, tokens[idx]).token_type == NoValueToken.Type.OPEN_BRACKET):
        return parse_expression(tokens, idx, result)
    else:
        return handle_atom(tokens, idx, result)


def run_translator(tokens: list[Token]) -> str:
    idx: int = 0
    result: Program = Program()

    while idx < len(tokens):
        token: Token = tokens[idx]
        parse_s_expression(tokens, idx, result)


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
