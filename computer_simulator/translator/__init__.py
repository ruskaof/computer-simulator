from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC


@dataclass
class ProgramChar:
    char: str
    line: int
    column: int


@dataclass
class Token(ABC):
    starting_char: ProgramChar


@dataclass
class NoValueToken(Token):
    class Type(Enum):
        IF: int = auto()
        OPEN_BRACKET: int = auto()
        CLOSE_BRACKET: int = auto()
        BINOP_PLUS: int = auto()
        BINOP_MINUS: int = auto()
        BINOP_EQUAL: int = auto()
        TRUE: int = auto()
        FALSE: int = auto()

    token_type: Type

    def __init__(self, token_type: Type, starting_char: ProgramChar) -> None:
        super().__init__(starting_char)
        self.token_type = token_type
        self.starting_char = starting_char


@dataclass
class ValueToken(Token):
    class Type(Enum):
        INT: int = auto()
        STRING: int = auto()
        IDENTIFIER: int = auto()

    token_type: Type
    value: str

    def __init__(self, token_type: Type, value: str, starting_char: ProgramChar) -> None:
        super().__init__(starting_char)
        self.token_type = token_type
        self.value = value
        self.starting_char = starting_char


class TranslatorException(Exception):
    bad_program_char: ProgramChar
    expected: str

    def __init__(self, bad_program_char: ProgramChar, expected: str) -> None:
        self.bad_program_char = bad_program_char
        self.expected = expected

    def __str__(self) -> str:
        return (f"Expected {self.expected} at line {self.bad_program_char.line}, column {self.bad_program_char.column}."
                f" Got {self.bad_program_char.char}")
