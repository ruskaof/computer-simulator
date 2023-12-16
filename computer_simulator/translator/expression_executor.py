import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from computer_simulator.isa import Opcode, ArgType
from computer_simulator.isa import Operation, Arg
from computer_simulator.translator import Token, STATIC_MEMORY_SIZE

EXPECTED_IDENTIFIER = "Expected identifier"
DEFAULT_WORD = 0
NUMBER_OFFSET_IN_UTF8 = 48

@dataclass
class Variable:
    name: Optional[str]
    died: bool


@dataclass
class PascalStringHeader:
    length: int


@dataclass
class StringCharacter:
    value: str


class AccumValueType(Enum):
    INT: str = "INT"
    STRING: str = "STRING"


class StackValue:
    class Type(Enum):
        INT: str = "INT"
        STRING_ADDR: str = "STRING"
        RETURN_ADDR: str = "RETURN_ADDR"

    def __init__(self, value: int, value_type: Type, name: Optional[str] = None):
        self.value: int = value
        self.value_type: StackValue.Type = value_type
        self.name: Optional[str] = name


class Program:
    def __init__(self):
        # only for strings
        self.memory: list[int | Operation] = [0 for _ in range(STATIC_MEMORY_SIZE)]
        self.current_stack: list[StackValue] = []
        self.memory_used = 1

    def load(self, value: int) -> None:
        self.memory.append(Operation(Opcode.LD, Arg(value, ArgType.DIRECT)))

    # allocates variable on top of stack
    def alloc_variable(self, name: Optional[str] = None) -> None:
        self.memory.append(Operation(Opcode.PUSH, None))
        self.current_stack.append(StackValue(len(self.memory), StackValue.Type.INT, name))

    def alloc_string(self, value: str) -> int:
        self.memory[self.memory_used] = len(value)
        self.memory_used += 1
        self.memory.append(Operation(Opcode.LD, Arg(len(value), ArgType.DIRECT)))
        self.memory.append(Operation(Opcode.ST, Arg(len(self.memory) - 1, ArgType.ADDRESS)))
        for char in value:
            self.memory[self.memory_used] = ord(char)
            self.memory_used += 1
            self.memory.append(Operation(Opcode.LD, Arg(ord(char), ArgType.DIRECT)))
            self.memory.append(Operation(Opcode.ST, Arg(len(self.memory) - 1, ArgType.ADDRESS)))
        return len(self.memory) - len(value) - 1

    def get_var_sp_offset(self, name: str) -> int:
        for i in range(len(self.current_stack) - 1, -1, -1):
            if self.current_stack[i].name == name:
                return len(self.current_stack) - i

    def to_machine_code(self) -> str:
        memory = []
        self.operation_to_dict(0, memory)

        for i in range(len(self.memory) - STATIC_MEMORY_SIZE):
            address = i + STATIC_MEMORY_SIZE
            self.operation_to_dict(address, memory)

        return json.dumps({"memory": memory}, indent=4)

    def operation_to_dict(self, address, memory):
        if self.memory[address].arg is None:
            memory.append(
                {
                    "opcode": self.memory[address].opcode.value,
                    "address": address,
                }
            )
        else:
            arg = self.memory[address].arg

            memory.append(
                {
                    "opcode": self.memory[address].opcode.value,
                    "arg": {
                        "value": arg.value,
                        "type": arg.type.value,
                    },
                    "address": address,
                },
            )


def exec_binop(op: str, program: Program) -> None:
    if op == "+":
        program.memory.append(Operation(Opcode.ADD, Arg(1, ArgType.STACK_OFFSET)))
    elif op == "=":
        program.memory.append(Operation(Opcode.EQ, Arg(1, ArgType.STACK_OFFSET)))
    elif op == "%":
        program.memory.append(Operation(Opcode.MOD, Arg(1, ArgType.STACK_OFFSET)))
    elif op == "/":
        program.memory.append(Operation(Opcode.DIV, Arg(1, ArgType.STACK_OFFSET)))


def _is_expression_start(tokens: list[Token], idx: int) -> bool:
    return tokens[idx].token_type in (
        Token.Type.OPEN_BRACKET,
        Token.Type.INT,
        Token.Type.BINOP,
        Token.Type.IF,
        Token.Type.SETQ,
        Token.Type.IDENTIFIER,
        Token.Type.STRING,
    )


def get_expr_end_idx(tokens: list[Token], idx: int, started_with_open_bracket: bool) -> int:
    if tokens[idx].token_type == Token.Type.CLOSE_BRACKET and started_with_open_bracket:
        return idx + 1
    elif not started_with_open_bracket:
        return idx
    else:
        raise RuntimeError(f"Expected close bracket in index {idx}")


def translate_expression(tokens: list[Token], idx: int, result: Program) -> int:
    if idx >= len(tokens):
        return idx
    elif not _is_expression_start(tokens, idx):
        raise RuntimeError("Expected expression")

    started_with_open_bracket: bool = False
    if tokens[idx].token_type == Token.Type.OPEN_BRACKET:
        idx += 1
        started_with_open_bracket = True

    if tokens[idx].token_type == Token.Type.INT:
        result.load(int(tokens[idx].value))
        return get_expr_end_idx(tokens, idx + 1, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.BINOP:
        first_expr_end_idx: int = translate_expression(tokens, idx + 1, result)
        result.alloc_variable()
        result.memory.append(Operation(Opcode.ST, Arg(1, ArgType.STACK_OFFSET)))
        second_expr_end_idx: int = translate_expression(tokens, first_expr_end_idx, result)
        exec_binop(tokens[idx].value, result)
        result.current_stack.pop()
        return get_expr_end_idx(tokens, second_expr_end_idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.IF:
        condition_end_idx: int = translate_expression(tokens, idx + 1, result)
        je_idx: int = len(result.memory)
        result.memory.append(Operation(Opcode.JZ, None))
        true_branch_end_idx: int = translate_expression(tokens, condition_end_idx, result)
        jmp_idx: int = len(result.memory)
        result.memory.append(Operation(Opcode.JMP, None))
        false_branch_memory_idx: int = len(result.memory)
        false_branch_end_idx: int = translate_expression(tokens, true_branch_end_idx, result)
        result.memory[je_idx].arg = Arg(false_branch_memory_idx, ArgType.ADDRESS)
        result.memory[jmp_idx].arg = Arg(len(result.memory), ArgType.ADDRESS)
        return get_expr_end_idx(tokens, false_branch_end_idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.SETQ:
        if tokens[idx + 1].token_type != Token.Type.IDENTIFIER:
            raise RuntimeError(EXPECTED_IDENTIFIER)
        expr_end_idx: int = translate_expression(tokens, idx + 2, result)
        result.alloc_variable(tokens[idx + 1].value)
        result.memory.append(Operation(Opcode.ST, Arg(1, ArgType.STACK_OFFSET)))
        return get_expr_end_idx(tokens, expr_end_idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.IDENTIFIER:
        result.memory.append(
            Operation(Opcode.LD, Arg(result.get_var_sp_offset(tokens[idx].value), ArgType.STACK_OFFSET)))
        return get_expr_end_idx(tokens, idx + 1, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.PROGN:
        idx += 1
        while tokens[idx].token_type == Token.Type.OPEN_BRACKET:
            idx = translate_expression(tokens, idx, result)
        return get_expr_end_idx(tokens, idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.STRING:
        string_addr = result.alloc_string(tokens[idx].value)
        result.load(string_addr)
        return get_expr_end_idx(tokens, idx + 1, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.PRINT_INT:
        idx = translate_expression(tokens, idx + 1, result)
        result.memory.append(Operation(Opcode.ADD, Arg(NUMBER_OFFSET_IN_UTF8, ArgType.DIRECT)))
        result.memory.append(Operation(Opcode.OUT, None))
        return get_expr_end_idx(tokens, idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.PRINT_STRING:
        idx = translate_expression(tokens, idx + 1, result)

        result.alloc_variable("#str_p")
        # save string pointer
        result.memory.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#str_p"), ArgType.STACK_OFFSET)))

        # save string size:
        # load string size to ac
        result.memory.append(Operation(Opcode.LD_BY_AC, None))
        # store string size
        result.alloc_variable("#str_size")
        result.memory.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#str_size"), ArgType.STACK_OFFSET)))

        # init index
        result.alloc_variable("#i")
        result.memory.append(Operation(Opcode.LD, Arg(0, ArgType.DIRECT)))
        result.memory.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))

        loop_start_idx: int = len(result.memory)
        # compare index with string size:
        # load index
        result.memory.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))
        result.memory.append(Operation(Opcode.EQ, Arg(result.get_var_sp_offset("#str_size"), ArgType.STACK_OFFSET)))

        jnz_idx: int = len(result.memory)
        # jump if index == string size
        result.memory.append(Operation(Opcode.JNZ, None))

        # load string pointer
        result.memory.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#str_p"), ArgType.STACK_OFFSET)))
        result.memory.append(Operation(Opcode.ADD, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))

        result.memory.append(Operation(Opcode.ADD, Arg(1, ArgType.DIRECT)))

        # load char
        result.memory.append(Operation(Opcode.LD_BY_AC, None))
        result.memory.append(Operation(Opcode.OUT, None))

        # increment index
        result.memory.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))
        result.memory.append(Operation(Opcode.ADD, Arg(1, ArgType.DIRECT)))
        result.memory.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))

        # jump to compare index with string size
        result.memory.append(Operation(Opcode.JMP, Arg(loop_start_idx, ArgType.ADDRESS)))
        result.memory[jnz_idx].arg = Arg(len(result.memory), ArgType.ADDRESS)

        result.current_stack.pop()  # i
        result.current_stack.pop()  # str_size
        result.current_stack.pop()  # str_p
        return get_expr_end_idx(tokens, idx, started_with_open_bracket)


def translate_program(tokens: list[Token], result: Program) -> None:
    result.memory[0] = Operation(Opcode.JMP, Arg(STATIC_MEMORY_SIZE + 1, ArgType.ADDRESS))
    translate_expression(tokens, 0, result)
    result.memory.append(Operation(Opcode.HLT, None))
