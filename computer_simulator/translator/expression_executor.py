from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from computer_simulator.isa import Arg, ArgType, Opcode, Operation
from computer_simulator.translator import Token

EXPECTED_IDENTIFIER = "Expected identifier"
STATIC_MEMORY_SIZE = 512
DEFAULT_WORD = 0
NUMBER_OFFSET_IN_UTF8 = 48
STRING_ALLOC_SIZE = 32
SERVICE_VARIABLE_ADDRESS = 1


@dataclass
class Variable:
    name: str | None
    died: bool


class AccumValueType(Enum):
    INT: str = "INT"
    STRING: str = "STRING"


class StackValue:
    class Type(Enum):
        INT: str = "INT"
        STRING_ADDR: str = "STRING"
        RETURN_ADDR: str = "RETURN_ADDR"

    def __init__(self, value: int, value_type: Type, name: str | None = None):
        self.value: int = value
        self.value_type: StackValue.Type = value_type
        self.name: str | None = name


class Program:
    def __init__(self):
        # only for strings
        self.memory: list[int | Operation] = [0 for _ in range(STATIC_MEMORY_SIZE)]
        self.memory[0] = Operation(Opcode.JMP, Arg(STATIC_MEMORY_SIZE, ArgType.ADDRESS))
        self.memory_used: int = 2
        self.current_stack: list[StackValue] = []

    def load(self, value: int) -> None:
        self.memory.append(Operation(Opcode.LD, Arg(value, ArgType.DIRECT)))

    # allocates variable on top of stack
    def push_var_to_stack(self, name: str | None = None) -> None:
        self.memory.append(Operation(Opcode.PUSH, None))
        self.current_stack.append(StackValue(len(self.memory), StackValue.Type.INT, name))

    def pop_var_from_stack(self) -> None:
        self.memory.append(Operation(Opcode.POP, None))
        self.current_stack.pop()

    def alloc_string(self, value: str) -> int:
        address = self.memory_used
        self.memory[self.memory_used] = len(value)
        self.memory.append(Operation(Opcode.LD, Arg(len(value), ArgType.DIRECT)))
        self.memory.append(Operation(Opcode.ST, Arg(self.memory_used, ArgType.ADDRESS)))
        self.memory_used += 1
        for char in value:
            self.memory[self.memory_used] = ord(char)
            self.memory.append(Operation(Opcode.LD, Arg(ord(char), ArgType.DIRECT)))
            self.memory.append(Operation(Opcode.ST, Arg(self.memory_used, ArgType.ADDRESS)))
            self.memory_used += 1
        return address

    def alloc_string_of_size(self, size: int) -> int:
        address = self.memory_used
        self.memory[self.memory_used] = size
        self.memory.append(Operation(Opcode.LD, Arg(size, ArgType.DIRECT)))
        self.memory.append(Operation(Opcode.ST, Arg(self.memory_used, ArgType.ADDRESS)))
        self.memory_used += 1
        for _ in range(size):
            self.memory[self.memory_used] = 0
            self.memory_used += 1
        return address

    def get_var_sp_offset(self, name: str) -> int:
        for i in range(len(self.current_stack) - 1, -1, -1):
            if self.current_stack[i].name == name:
                return len(self.current_stack) - i
        return None

    def to_machine_code(self) -> str:
        memory = []

        for i in range(len(self.memory)):
            if isinstance(self.memory[i], Operation):
                if self.memory[i].arg is None:
                    memory.append(
                        {
                            "opcode": self.memory[i].opcode.value,
                            "address": i,
                        }
                    )
                else:
                    arg = self.memory[i].arg

                    memory.append(
                        {
                            "opcode": self.memory[i].opcode.value,
                            "arg": {
                                "value": arg.value,
                                "type": arg.arg_type.value,
                            },
                            "address": i,
                        },
                    )
        return json.dumps({"memory": memory}, indent=4)


def exec_binop(op: str, program: Program) -> None:
    if op == "+":
        program.memory.append(Operation(Opcode.ADD, Arg(1, ArgType.STACK_OFFSET)))
    elif op == "=":
        program.memory.append(Operation(Opcode.EQ, Arg(1, ArgType.STACK_OFFSET)))
    elif op == "%":
        program.memory.append(Operation(Opcode.MOD, Arg(1, ArgType.STACK_OFFSET)))
    elif op == "/":
        program.memory.append(Operation(Opcode.DIV, Arg(1, ArgType.STACK_OFFSET)))
    elif op == "<":
        program.memory.append(Operation(Opcode.LT, Arg(1, ArgType.STACK_OFFSET)))
    elif op == ">":
        program.memory.append(Operation(Opcode.GT, Arg(1, ArgType.STACK_OFFSET)))
    else:
        raise RuntimeError(f"Unexpected binop: {op}")


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
        raise RuntimeError(f"Expected close bracket, got {tokens[idx]}")


def seek_end_of_expression(tokens: list[Token], idx: int) -> int:
    if idx >= len(tokens):
        return idx
    elif tokens[idx].token_type == Token.Type.OPEN_BRACKET:
        idx += 1
        while tokens[idx].token_type != Token.Type.CLOSE_BRACKET:
            idx = seek_end_of_expression(tokens, idx)
        return idx + 1
    elif tokens[idx].token_type == Token.Type.CLOSE_BRACKET:
        return idx + 1
    else:
        return idx + 1


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
        first_expr_end_idx: int = seek_end_of_expression(tokens, idx + 1)
        second_expr_end_idx: int = translate_expression(tokens, first_expr_end_idx, result)
        result.push_var_to_stack()
        translate_expression(tokens, idx + 1, result)
        exec_binop(tokens[idx].value, result)
        result.pop_var_from_stack()
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

        varname: str = tokens[idx + 1].value
        var_sp_offset: int = result.get_var_sp_offset(varname)
        if var_sp_offset is None:
            result.push_var_to_stack(varname)
            var_sp_offset = result.get_var_sp_offset(varname)

        result.memory.append(Operation(Opcode.ST, Arg(var_sp_offset, ArgType.STACK_OFFSET)))
        return get_expr_end_idx(tokens, expr_end_idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.IDENTIFIER:
        result.memory.append(
            Operation(Opcode.LD, Arg(result.get_var_sp_offset(tokens[idx].value), ArgType.STACK_OFFSET))
        )
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
    elif tokens[idx].token_type == Token.Type.PRINT_CHAR:
        idx = translate_expression(tokens, idx + 1, result)
        result.memory.append(Operation(Opcode.OUT, None))
        return get_expr_end_idx(tokens, idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.PRINT_STRING:
        idx = translate_expression(tokens, idx + 1, result)

        result.push_var_to_stack("#str_p")
        # save string pointer
        result.memory.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#str_p"), ArgType.STACK_OFFSET)))

        # save string size:
        # load string size to ac
        result.memory.append(Operation(Opcode.ST, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.ADDRESS)))
        result.memory.append(Operation(Opcode.LD, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.INDIRECT)))

        # store string size
        result.push_var_to_stack("#str_size")
        result.memory.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#str_size"), ArgType.STACK_OFFSET)))

        # init index
        result.push_var_to_stack("#i")
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
        result.memory.append(Operation(Opcode.ST, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.ADDRESS)))
        result.memory.append(Operation(Opcode.LD, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.INDIRECT)))
        result.memory.append(Operation(Opcode.OUT, None))

        # increment index
        result.memory.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))
        result.memory.append(Operation(Opcode.ADD, Arg(1, ArgType.DIRECT)))
        result.memory.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))

        # jump to compare index with string size
        result.memory.append(Operation(Opcode.JMP, Arg(loop_start_idx, ArgType.ADDRESS)))
        result.memory[jnz_idx].arg = Arg(len(result.memory), ArgType.ADDRESS)

        result.pop_var_from_stack()  # i
        result.pop_var_from_stack()  # str_size
        result.pop_var_from_stack()  # str_p
        return get_expr_end_idx(tokens, idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.READ_STRING:
        if tokens[idx + 1].token_type != Token.Type.IDENTIFIER:
            raise RuntimeError(EXPECTED_IDENTIFIER)
        varname = tokens[idx + 1].value

        var_sp_offset = result.get_var_sp_offset(varname)
        if var_sp_offset is None:
            result.push_var_to_stack(varname)

        # alloc string
        string_addr = result.alloc_string_of_size(STRING_ALLOC_SIZE)
        result.memory.append(Operation(Opcode.LD, Arg(string_addr, ArgType.DIRECT)))

        # save string pointer
        result.push_var_to_stack("#str_p")

        # index
        result.memory.append(Operation(Opcode.LD, Arg(1, ArgType.DIRECT)))
        result.push_var_to_stack("#i")

        # char
        result.push_var_to_stack("#char")

        # cycle start
        cycle_start_idx = len(result.memory)

        # read char
        result.memory.append(Operation(Opcode.IN, None))
        result.memory.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#char"), ArgType.STACK_OFFSET)))

        # if char is 0, then break
        result.memory.append(Operation(Opcode.EQ, Arg(0, ArgType.DIRECT)))
        jz_idx = len(result.memory)
        result.memory.append(Operation(Opcode.JNZ, None))

        # save char by index
        result.memory.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#str_p"), ArgType.STACK_OFFSET)))
        result.memory.append(Operation(Opcode.ADD, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))
        result.memory.append(Operation(Opcode.ST, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.ADDRESS)))
        result.memory.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#char"), ArgType.STACK_OFFSET)))
        result.memory.append(Operation(Opcode.ST, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.INDIRECT)))

        # increment index
        result.memory.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))
        result.memory.append(Operation(Opcode.ADD, Arg(1, ArgType.DIRECT)))
        result.memory.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))

        # jump to cycle start
        result.memory.append(Operation(Opcode.JMP, Arg(cycle_start_idx, ArgType.ADDRESS)))
        result.memory[jz_idx].arg = Arg(len(result.memory), ArgType.ADDRESS)

        # save string size
        result.memory.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#str_p"), ArgType.STACK_OFFSET)))
        result.memory.append(Operation(Opcode.ST, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.ADDRESS)))
        result.memory.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))
        result.memory.append(Operation(Opcode.SUB, Arg(1, ArgType.DIRECT)))
        result.memory.append(Operation(Opcode.ST, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.INDIRECT)))

        # save string pointer to variable
        result.memory.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#str_p"), ArgType.STACK_OFFSET)))
        result.memory.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset(varname), ArgType.STACK_OFFSET)))

        result.pop_var_from_stack()  # i
        result.pop_var_from_stack()  # char
        result.pop_var_from_stack()  # str_p

        result.memory.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset(varname), ArgType.STACK_OFFSET)))

        return get_expr_end_idx(tokens, idx + 2, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.WHILE:
        loop_start_idx = len(result.memory)
        condition_end_idx = translate_expression(tokens, idx + 1, result)
        jz_idx = len(result.memory)
        result.memory.append(Operation(Opcode.JZ, None))
        body_end_idx = translate_expression(tokens, condition_end_idx, result)
        result.memory.append(Operation(Opcode.JMP, Arg(loop_start_idx, ArgType.ADDRESS)))
        result.memory[jz_idx].arg = Arg(len(result.memory), ArgType.ADDRESS)

        return get_expr_end_idx(tokens, body_end_idx, started_with_open_bracket)
    return None


def translate_program(tokens: list[Token], result: Program) -> None:
    translate_expression(tokens, 0, result)
    result.memory.append(Operation(Opcode.HLT, None))
