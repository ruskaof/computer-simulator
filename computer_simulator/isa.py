from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


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


class ArgType(Enum):
    DIRECT: str = "DIRECT"
    ADDRESS: str = "ADDRESS"
    INDIRECT: str = "INDIRECT"
    STACK_OFFSET: str = "STACK_OFFSET"


@dataclass
class Arg:
    value: int
    type: ArgType


@dataclass
class Operation:
    opcode: Opcode
    arg: Optional[Arg]
