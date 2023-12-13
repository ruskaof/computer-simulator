from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC
from typing import NewType, Optional


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
        PRINT_INT: int = auto()
        PRINT_STRING: int = auto()
        PROGN: int = auto()

    token_type: Type
    value: str
