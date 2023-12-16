from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class Token(ABC):
    class Type(Enum):
        IF: int = auto()
        OPEN_BRACKET: int = auto()
        CLOSE_BRACKET: int = auto()
        BINOP: int = auto()
        BOOLEAN: int = auto()
        INT: int = auto()
        STRING: int = auto()
        SETQ: int = auto()
        IDENTIFIER: int = auto()
        DEFUN: int = auto()
        PRINT_CHAR: int = auto()
        PRINT_STRING: int = auto()
        PROGN: int = auto()
        READ_STRING: int = auto()
        WHILE: int = auto()

    token_type: Type
    value: str
