from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Opcode(Enum):
    LD: str = "LD"
    ST: str = "ST"
    ADD: str = "ADD"
    SUB: str = "SUB"
    LT: str = "LT"
    GT: str = "GT"
    EQ: str = "EQ"
    MOD: str = "MOD"
    DIV: str = "DIV"
    MUL: str = "MUL"
    JZ: str = "JZ"
    JNZ: str = "JNZ"
    JMP: str = "JMP"
    POP: str = "POP"
    PUSH: str = "PUSH"
    CALL: str = "CALL"
    RET: str = "RET"
    IN: str = "IN"
    OUT: str = "OUT"
    HLT: str = "HLT"

    def __str__(self) -> str:
        return self.value


class ArgType(Enum):
    DIRECT: str = "DIRECT"
    ADDRESS: str = "ADDRESS"
    INDIRECT: str = "INDIRECT"
    STACK_OFFSET: str = "STACK_OFFSET"

    def __str__(self) -> str:
        return self.value


@dataclass
class Arg:
    value: int
    arg_type: ArgType

    def __str__(self) -> str:
        return f"{self.value} ({self.arg_type})"


@dataclass
class Operation:
    opcode: Opcode
    arg: Arg | None
    comment: str | None = None

    def __str__(self) -> str:
        r = f"{self.opcode}"
        if self.arg:
            r += f" {self.arg}"
        if self.comment:
            r += f" ({self.comment})"
        return f"Operation({r})"

