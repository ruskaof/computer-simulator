from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, cast

from computer_simulator.isa import Arg, ArgType, Instruction, Opcode
from computer_simulator.translator import Token
from computer_simulator.translator.errors import InvalidSymbolsError

EXPECTED_IDENTIFIER = "Expected identifier"
STATIC_MEMORY_SIZE = 512
DEFAULT_WORD = 0
STRING_ALLOC_SIZE = 32
SERVICE_VAR_ADDR = 1


@dataclass
class Function:
    address: int
    setq_count: int


@dataclass
class StackValue:
    value: int
    name: str | None


class Program:
    def __init__(self):
        # only for strings
        self.memory: list[int | Instruction] = [0 for _ in range(STATIC_MEMORY_SIZE)]
        self.memory_used: int = 0
        self.current_stack: list[StackValue] = []
        self.functions: dict[str, int] = {}
        self.current_block_setq_count: int = 0

    def load(self, value: int) -> None:
        self.memory.append(Instruction(Opcode.LD, Arg(value, ArgType.DIRECT)))

    # allocates variable on top of stack
    def push_var_to_stack(self, name: str | None = None) -> None:
        self.memory.append(Instruction(Opcode.PUSH, arg=None, comment=f"Push var {name}"))
        self.current_stack.append(StackValue(len(self.memory), name))

    def resolve_stack_var(self, name: str) -> None:
        self.current_stack.append(StackValue(len(self.memory), name))

    def unresolve_stack_var(self, name: str) -> None:
        for i in range(len(self.current_stack) - 1, -1, -1):
            if self.current_stack[i].name == name:
                self.current_stack.pop(i)
                return
        raise InvalidSymbolsError(got=name, expected="known variable, but got unknown")

    def pop_var_from_stack(self, comment: str | None = None) -> None:
        self.memory.append(Instruction(Opcode.POP, None, comment))
        self.current_stack.pop()

    def alloc_string(self, value: str) -> int:
        address = self.memory_used
        self.memory[self.memory_used] = len(value)
        self.memory.append(Instruction(Opcode.LD, Arg(len(value), ArgType.DIRECT)))
        self.memory.append(Instruction(Opcode.ST, Arg(self.memory_used, ArgType.ADDRESS)))
        self.memory_used += 1
        for char in value:
            self.memory[self.memory_used] = ord(char)
            self.memory.append(Instruction(Opcode.LD, Arg(ord(char), ArgType.DIRECT)))
            self.memory.append(Instruction(Opcode.ST, Arg(self.memory_used, ArgType.ADDRESS)))
            self.memory_used += 1
        return address

    def alloc_string_of_size(self, size: int) -> int:
        address = self.memory_used
        self.memory[self.memory_used] = size
        self.memory.append(Instruction(Opcode.LD, Arg(size, ArgType.DIRECT)))
        self.memory.append(Instruction(Opcode.ST, Arg(self.memory_used, ArgType.ADDRESS)))
        self.memory_used += 1
        for _ in range(size):
            self.memory[self.memory_used] = 0
            self.memory_used += 1
        return address

    def get_var_sp_offset(self, name: str) -> int | None:
        for i in range(len(self.current_stack) - 1, -1, -1):
            if self.current_stack[i].name == name:
                return len(self.current_stack) - i - 1
        return None

    def to_machine_code(self) -> str:
        memory = []

        for i in range(len(self.memory)):
            if isinstance(self.memory[i], Instruction):
                instruction = cast(Instruction, self.memory[i])
                instruction_dict = {"opcode": instruction.opcode.value, "address": i}

                if instruction.arg is not None:
                    instruction_dict["arg"] = {
                        "value": instruction.arg.value,
                        "type": instruction.arg.arg_type.value,
                    }

                if instruction.comment is not None:
                    instruction_dict["comment"] = instruction.comment

                memory.append(instruction_dict)
        return json.dumps({"memory": memory}, indent=4)


BINOP_OPCODE: dict[str, Opcode] = {
    "+": Opcode.ADD,
    "=": Opcode.EQ,
    "%": Opcode.MOD,
    "/": Opcode.DIV,
    "<": Opcode.LT,
    ">": Opcode.GT,
    "*": Opcode.MUL,
}


def exec_binop(op: str, program: Program) -> None:
    program.memory.append(Instruction(BINOP_OPCODE[op], Arg(0, ArgType.STACK_OFFSET)))


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
    if not started_with_open_bracket:
        return idx

    raise InvalidSymbolsError(got=tokens[idx], expected="close bracket")


def seek_end_of_expression(tokens: list[Token], idx: int) -> int:
    if idx >= len(tokens):
        return idx
    if tokens[idx].token_type == Token.Type.OPEN_BRACKET:
        idx += 1
        while tokens[idx].token_type != Token.Type.CLOSE_BRACKET:
            idx = seek_end_of_expression(tokens, idx)
        return idx + 1
    return idx + 1


def get_args_of_func(tokens: list[Token], idx: int) -> tuple[list[str], int]:
    if tokens[idx].token_type != Token.Type.OPEN_BRACKET:
        raise InvalidSymbolsError(got=tokens[idx], expected="open bracket")
    idx += 1
    passed_args: list[str] = []
    while tokens[idx].token_type != Token.Type.CLOSE_BRACKET:
        if tokens[idx].token_type != Token.Type.IDENTIFIER:
            raise RuntimeError(EXPECTED_IDENTIFIER)
        passed_args.append(tokens[idx].value)
        idx += 1
    return passed_args, idx + 1


def pass_args_to_func(tokens: list[Token], idx: int, result: Program) -> tuple[int, int]:
    args_n = 0
    while tokens[idx].token_type != Token.Type.CLOSE_BRACKET:
        translate_expression(tokens, idx, result)
        result.memory.append(Instruction(Opcode.PUSH, None, "Push arg"))
        idx += 1
        args_n += 1
    return idx, args_n


def handle_token_int(tokens: list[Token], idx: int, result: Program, started_with_open_bracket: bool) -> int:
    result.load(int(tokens[idx].value))
    return get_expr_end_idx(tokens, idx + 1, started_with_open_bracket)


def handle_token_binop(tokens: list[Token], idx: int, result: Program, started_with_open_bracket: bool) -> int:
    first_expr_end_idx: int = seek_end_of_expression(tokens, idx + 1)
    second_expr_end_idx: int = translate_expression(tokens, first_expr_end_idx, result)
    result.push_var_to_stack("#binop result")
    translate_expression(tokens, idx + 1, result)
    exec_binop(tokens[idx].value, result)
    result.pop_var_from_stack()
    return get_expr_end_idx(tokens, second_expr_end_idx, started_with_open_bracket)


def handle_token_if(tokens: list[Token], idx: int, result: Program, started_with_open_bracket: bool) -> int:
    condition_end_idx: int = translate_expression(tokens, idx + 1, result)
    je_idx: int = len(result.memory)
    result.memory.append(Instruction(Opcode.JZ, None))
    true_branch_end_idx: int = translate_expression(tokens, condition_end_idx, result)
    jmp_idx: int = len(result.memory)
    result.memory.append(Instruction(Opcode.JMP, None))
    false_branch_memory_idx: int = len(result.memory)
    false_branch_end_idx: int = translate_expression(tokens, true_branch_end_idx, result)
    cast(Instruction, result.memory[je_idx]).arg = Arg(false_branch_memory_idx, ArgType.ADDRESS)
    cast(Instruction, result.memory[jmp_idx]).arg = Arg(len(result.memory), ArgType.ADDRESS)
    return get_expr_end_idx(tokens, false_branch_end_idx, started_with_open_bracket)


def handle_token_setq(tokens: list[Token], idx: int, result: Program, started_with_open_bracket: bool) -> int:
    if tokens[idx + 1].token_type != Token.Type.IDENTIFIER:
        raise RuntimeError(EXPECTED_IDENTIFIER)
    expr_end_idx: int = translate_expression(tokens, idx + 2, result)

    varname: str = tokens[idx + 1].value
    var_sp_offset: int | None = result.get_var_sp_offset(varname)
    if var_sp_offset is None:
        result.push_var_to_stack(varname)
        var_sp_offset = result.get_var_sp_offset(varname)
        result.current_block_setq_count += 1

    result.memory.append(Instruction(Opcode.ST, Arg(cast(int, var_sp_offset), ArgType.STACK_OFFSET)))
    return get_expr_end_idx(tokens, expr_end_idx, started_with_open_bracket)


def handle_token_identifier(tokens: list[Token], idx: int, result: Program, started_with_open_bracket: bool) -> int:
    if tokens[idx].value in result.functions:
        args_end_idx, args_n = pass_args_to_func(tokens, idx + 1, result)
        result.memory.append(Instruction(Opcode.CALL, Arg(result.functions[tokens[idx].value], ArgType.ADDRESS)))
        for arg in range(args_n):
            result.memory.append(Instruction(Opcode.POP, None, f"Pop arg {arg}"))
        return get_expr_end_idx(tokens, args_end_idx, started_with_open_bracket)

    result.memory.append(
        Instruction(Opcode.LD, Arg(cast(int, result.get_var_sp_offset(tokens[idx].value)), ArgType.STACK_OFFSET))
    )
    return get_expr_end_idx(tokens, idx + 1, started_with_open_bracket)


def handle_token_progn(tokens: list[Token], idx: int, result: Program, started_with_open_bracket: bool) -> int:
    idx += 1
    while tokens[idx].token_type == Token.Type.OPEN_BRACKET:
        idx = translate_expression(tokens, idx, result)
    return get_expr_end_idx(tokens, idx, started_with_open_bracket)


def handle_token_string(tokens: list[Token], idx: int, result: Program, started_with_open_bracket: bool) -> int:
    string_addr = result.alloc_string(tokens[idx].value)
    result.load(string_addr)
    return get_expr_end_idx(tokens, idx + 1, started_with_open_bracket)


def handle_token_print_char(tokens: list[Token], idx: int, result: Program, started_with_open_bracket: bool) -> int:
    idx = translate_expression(tokens, idx + 1, result)
    result.memory.append(Instruction(Opcode.OUT, None))
    return get_expr_end_idx(tokens, idx, started_with_open_bracket)


def handle_token_print_string(tokens: list[Token], idx: int, result: Program, started_with_open_bracket: bool) -> int:
    idx = translate_expression(tokens, idx + 1, result)
    str_pointer_varname = "#str_p"

    result.push_var_to_stack(str_pointer_varname)

    # save string size:
    # load string size to ac
    result.memory.append(Instruction(Opcode.ST, Arg(SERVICE_VAR_ADDR, ArgType.ADDRESS)))
    result.memory.append(
        Instruction(Opcode.LD, Arg(SERVICE_VAR_ADDR, ArgType.INDIRECT), "Load string size inside print_str")
    )

    # store string size
    result.push_var_to_stack("#str_size")

    # init index
    result.memory.append(Instruction(Opcode.LD, Arg(0, ArgType.DIRECT)))
    result.push_var_to_stack("#i")

    loop_start_idx: int = len(result.memory)
    # compare index with string size:
    # load index
    result.memory.append(Instruction(Opcode.LD, Arg(cast(int, result.get_var_sp_offset("#i")), ArgType.STACK_OFFSET)))
    result.memory.append(
        Instruction(Opcode.EQ, Arg(cast(int, result.get_var_sp_offset("#str_size")), ArgType.STACK_OFFSET))
    )

    jnz_idx: int = len(result.memory)
    # jump if index == string size
    result.memory.append(Instruction(Opcode.JNZ, None))

    # load string pointer
    result.memory.append(
        Instruction(Opcode.LD, Arg(cast(int, result.get_var_sp_offset(str_pointer_varname)), ArgType.STACK_OFFSET))
    )
    result.memory.append(Instruction(Opcode.ADD, Arg(cast(int, result.get_var_sp_offset("#i")), ArgType.STACK_OFFSET)))

    result.memory.append(Instruction(Opcode.ADD, Arg(1, ArgType.DIRECT)))

    # load char
    result.memory.append(Instruction(Opcode.ST, Arg(SERVICE_VAR_ADDR, ArgType.ADDRESS)))
    result.memory.append(Instruction(Opcode.LD, Arg(SERVICE_VAR_ADDR, ArgType.INDIRECT), "Load char inside print_str"))
    result.memory.append(Instruction(Opcode.OUT, None))

    # increment index
    increment_index(result)

    # jump to compare index with string size
    result.memory.append(Instruction(Opcode.JMP, Arg(loop_start_idx, ArgType.ADDRESS), "Jump to read str loop start"))
    cast(Instruction, result.memory[jnz_idx]).arg = Arg(len(result.memory), ArgType.ADDRESS)

    result.memory.append(
        Instruction(Opcode.LD, Arg(cast(int, result.get_var_sp_offset(str_pointer_varname)), ArgType.STACK_OFFSET))
    )

    result.pop_var_from_stack(comment="Pop #i used to print string")
    result.pop_var_from_stack(comment="Pop #str_size used to print string")
    result.pop_var_from_stack(comment="Pop #str_p used to print string")

    return get_expr_end_idx(tokens, idx, started_with_open_bracket)


def increment_index(result, index_name: str = "#i"):
    result.memory.append(Instruction(Opcode.LD, Arg(result.get_var_sp_offset(index_name), ArgType.STACK_OFFSET)))
    result.memory.append(Instruction(Opcode.ADD, Arg(1, ArgType.DIRECT)))
    result.memory.append(Instruction(Opcode.ST, Arg(result.get_var_sp_offset(index_name), ArgType.STACK_OFFSET)))


def handle_token_read_string(tokens: list[Token], idx: int, result: Program, started_with_open_bracket: bool) -> int:
    if tokens[idx + 1].token_type != Token.Type.IDENTIFIER:
        raise RuntimeError(EXPECTED_IDENTIFIER)
    varname = tokens[idx + 1].value

    var_sp_offset = result.get_var_sp_offset(varname)
    if var_sp_offset is None:
        result.push_var_to_stack(varname)

    # alloc string
    string_addr = result.alloc_string_of_size(STRING_ALLOC_SIZE)
    result.memory.append(Instruction(Opcode.LD, Arg(string_addr, ArgType.DIRECT)))

    str_pointer_varname = "#str_p"
    result.push_var_to_stack(str_pointer_varname)

    # index
    result.memory.append(Instruction(Opcode.LD, Arg(1, ArgType.DIRECT)))
    result.push_var_to_stack("#i")

    char_varname = "#char"
    result.push_var_to_stack(char_varname)

    # cycle start
    cycle_start_idx = len(result.memory)

    # read char
    result.memory.append(Instruction(Opcode.IN, None))
    result.memory.append(
        Instruction(Opcode.ST, Arg(cast(int, result.get_var_sp_offset(char_varname)), ArgType.STACK_OFFSET))
    )

    # if char is 0, then break
    result.memory.append(Instruction(Opcode.EQ, Arg(0, ArgType.DIRECT)))
    jz_idx = len(result.memory)
    result.memory.append(Instruction(Opcode.JNZ, None))

    # save char by index
    result.memory.append(
        Instruction(Opcode.LD, Arg(cast(int, result.get_var_sp_offset(str_pointer_varname)), ArgType.STACK_OFFSET))
    )
    result.memory.append(Instruction(Opcode.ADD, Arg(cast(int, result.get_var_sp_offset("#i")), ArgType.STACK_OFFSET)))
    result.memory.append(Instruction(Opcode.ST, Arg(SERVICE_VAR_ADDR, ArgType.ADDRESS)))
    result.memory.append(
        Instruction(Opcode.LD, Arg(cast(int, result.get_var_sp_offset(char_varname)), ArgType.STACK_OFFSET))
    )
    result.memory.append(Instruction(Opcode.ST, Arg(SERVICE_VAR_ADDR, ArgType.INDIRECT), "Save char by index"))

    increment_index(result)

    # jump to cycle start
    result.memory.append(Instruction(Opcode.JMP, Arg(cycle_start_idx, ArgType.ADDRESS)))
    cast(Instruction, result.memory[jz_idx]).arg = Arg(len(result.memory), ArgType.ADDRESS)

    # save string size
    result.memory.append(
        Instruction(Opcode.LD, Arg(cast(int, result.get_var_sp_offset(str_pointer_varname)), ArgType.STACK_OFFSET))
    )
    result.memory.append(Instruction(Opcode.ST, Arg(SERVICE_VAR_ADDR, ArgType.ADDRESS)))
    result.memory.append(Instruction(Opcode.LD, Arg(cast(int, result.get_var_sp_offset("#i")), ArgType.STACK_OFFSET)))
    result.memory.append(Instruction(Opcode.SUB, Arg(1, ArgType.DIRECT)))
    result.memory.append(Instruction(Opcode.ST, Arg(SERVICE_VAR_ADDR, ArgType.INDIRECT), "Save string size"))

    # save string pointer to variable
    result.memory.append(
        Instruction(Opcode.LD, Arg(cast(int, result.get_var_sp_offset(str_pointer_varname)), ArgType.STACK_OFFSET))
    )
    result.memory.append(
        Instruction(Opcode.ST, Arg(cast(int, result.get_var_sp_offset(varname)), ArgType.STACK_OFFSET))
    )

    result.pop_var_from_stack(comment="Pop #i used to read string")
    result.pop_var_from_stack(comment="Pop #char used to read string")
    result.pop_var_from_stack(comment="Pop #str_p used to read string")

    result.memory.append(
        Instruction(Opcode.LD, Arg(cast(int, result.get_var_sp_offset(varname)), ArgType.STACK_OFFSET))
    )

    return get_expr_end_idx(tokens, idx + 2, started_with_open_bracket)


def handle_token_while(tokens: list[Token], idx: int, result: Program, started_with_open_bracket: bool) -> int:
    loop_start_idx = len(result.memory)
    condition_end_idx = translate_expression(tokens, idx + 1, result)
    jz_idx = len(result.memory)
    result.memory.append(Instruction(Opcode.JZ, None))
    body_end_idx = translate_expression(tokens, condition_end_idx, result)
    result.memory.append(Instruction(Opcode.JMP, Arg(loop_start_idx, ArgType.ADDRESS)))
    cast(Instruction, result.memory[jz_idx]).arg = Arg(len(result.memory), ArgType.ADDRESS)

    result.memory.append(Instruction(Opcode.LD, Arg(0, ArgType.DIRECT), "Load 0 so that defun expression returns 0"))

    return get_expr_end_idx(tokens, body_end_idx, started_with_open_bracket)


def handle_token_defun(tokens: list[Token], idx: int, result: Program, started_with_open_bracket: bool) -> int:
    if tokens[idx + 1].token_type != Token.Type.IDENTIFIER:
        raise RuntimeError(EXPECTED_IDENTIFIER)

    jmp_idx = len(result.memory)
    result.memory.append(Instruction(Opcode.JMP, None))
    result.functions[tokens[idx + 1].value] = len(result.memory)

    # add variables to stack
    stack_variables, args_end_idx = get_args_of_func(tokens, idx + 2)
    for var in stack_variables:
        result.resolve_stack_var(var)
    result.resolve_stack_var("#ret_addr")

    outer_setq_count = result.current_block_setq_count
    result.current_block_setq_count = 0
    body_end_idx = translate_expression(tokens, args_end_idx, result)
    for _ in range(result.current_block_setq_count):
        result.memory.append(Instruction(Opcode.POP, None, f"Pop local var of function {tokens[idx + 1].value}"))
    result.current_block_setq_count = outer_setq_count

    result.memory.append(Instruction(Opcode.RET, None))

    cast(Instruction, result.memory[jmp_idx]).arg = Arg(len(result.memory), ArgType.ADDRESS)

    for var in stack_variables:
        result.unresolve_stack_var(var)

    result.memory.append(Instruction(Opcode.LD, Arg(0, ArgType.DIRECT), "Load 0 so that defun expression returns 0"))

    return get_expr_end_idx(tokens, body_end_idx, started_with_open_bracket)


TOKEN_HANDLERS: dict[Token.Type, Callable[[list[Token], int, Program, bool], int]] = {
    Token.Type.INT: handle_token_int,
    Token.Type.BINOP: handle_token_binop,
    Token.Type.IF: handle_token_if,
    Token.Type.SETQ: handle_token_setq,
    Token.Type.IDENTIFIER: handle_token_identifier,
    Token.Type.PROGN: handle_token_progn,
    Token.Type.STRING: handle_token_string,
    Token.Type.PRINT_CHAR: handle_token_print_char,
    Token.Type.PRINT_STRING: handle_token_print_string,
    Token.Type.READ_STRING: handle_token_read_string,
    Token.Type.WHILE: handle_token_while,
    Token.Type.DEFUN: handle_token_defun,
}


def translate_expression(tokens: list[Token], idx: int, result: Program) -> int:
    if idx >= len(tokens):
        return idx
    if not _is_expression_start(tokens, idx):
        raise InvalidSymbolsError(got=tokens[idx], expected="expression start")

    started_with_open_bracket: bool = False
    if tokens[idx].token_type == Token.Type.OPEN_BRACKET:
        idx += 1
        started_with_open_bracket = True

    return TOKEN_HANDLERS[tokens[idx].token_type](tokens, idx, result, started_with_open_bracket)


def translate_program(tokens: list[Token], result: Program) -> None:
    result.memory[0] = Instruction(Opcode.JMP, Arg(STATIC_MEMORY_SIZE, ArgType.ADDRESS), "Skip static memory")
    result.memory_used = 2  # for jmp and for service var
    translate_expression(tokens, 0, result)
    result.memory.append(Instruction(Opcode.HLT, None))
