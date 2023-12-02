from dataclasses import dataclass
from enum import Enum, auto


class Opcode(Enum):
    LD: int = auto()
    ST: int = auto()
    ADD: int = auto()
    SUB: int = auto()
    MUL: int = auto()
    DIV: int = auto()
    EQ: int = auto()
    JE: int = auto()
    JMP: int = auto()
    POP: int = auto()
    PUSH: int = auto()
    CALL: int = auto()
    RET: int = auto()


@dataclass
class Operation:
    class ArgType(Enum):
        DIRECT: int = auto()
        ADDRESS: int = auto()

    opcode: Opcode
    arg: int
    arg_type: ArgType
