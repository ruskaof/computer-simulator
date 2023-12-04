import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from computer_simulator.isa import Opcode
from computer_simulator.translator import Token

EXPECTED_IDENTIFIER = "Expected identifier"


class ArgType(Enum):
    DIRECT: str = "DIRECT"
    DATA_ADDRESS: str = "DATA_ADDRESS"
    INDIRECT_DATA_ADDRESS: str = "INDIRECT_DATA_ADDRESS"
    PROGRAM_ADDRESS: str = "PROGRAM_ADDRESS"


@dataclass
class Arg:
    value: int
    type: ArgType


@dataclass
class Operation:
    opcode: Opcode
    arg: Optional[Arg]


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


class Program:
    def __init__(self):
        self.data_memory: list[Variable | PascalStringHeader | StringCharacter] = []
        self.operations: list[Operation] = []
        self.accum_value_type: AccumValueType = AccumValueType.INT

    def load_int(self, value: int) -> None:
        self.accum_value_type = AccumValueType.INT
        self.operations.append(Operation(Opcode.LD, Arg(value, ArgType.DIRECT)))

    def load_string(self, addr: int) -> None:
        self.accum_value_type = AccumValueType.STRING
        self.operations.append(Operation(Opcode.LD, Arg(addr, ArgType.DIRECT)))

    def get_last_operation_index(self) -> int:
        return int(len(self.operations) - 1)

    def alloc_string(self, value: str) -> int:
        self.data_memory.append(PascalStringHeader(len(value)))
        self.operations.append(Operation(Opcode.LD, Arg(len(value), ArgType.DIRECT)))
        self.operations.append(Operation(Opcode.ST, Arg(len(self.data_memory) - 1, ArgType.DATA_ADDRESS)))
        for char in value:
            self.data_memory.append(StringCharacter(char))
            self.operations.append(Operation(Opcode.LD, Arg(ord(char), ArgType.DIRECT)))
            self.operations.append(Operation(Opcode.ST, Arg(len(self.data_memory) - 1, ArgType.DATA_ADDRESS)))
        return len(self.data_memory) - len(value) - 1

    def alloc_variable(self, name: Optional[str] = None) -> int:
        for i in range(len(self.data_memory)):
            if isinstance(self.data_memory[i], Variable) and self.data_memory[i].died:
                self.data_memory[i].died = False
                self.data_memory[i].name = name
                return i

        self.data_memory.append(Variable(name, False))
        return len(self.data_memory) - 1

    def remove_intermediate_variable(self, addr: int) -> None:
        self.data_memory[addr].died = True

    def get_variable_address(self, name: str) -> int:
        for i in range(len(self.data_memory)):
            if isinstance(self.data_memory[i], Variable) and self.data_memory[i].name == name:
                return i
        raise RuntimeError(f"Variable {name} not found")

    def to_machine_code(self) -> str:
        instructions_offset: int = len(self.data_memory)
        memory = []

        for memory_item in self.data_memory:
            if isinstance(memory_item, Variable):
                memory.append(0)
            elif isinstance(memory_item, PascalStringHeader):
                memory.append(0)
            elif isinstance(memory_item, StringCharacter):
                memory.append(0)
            else:
                raise RuntimeError("Unknown memory item")

        for operation in self.operations:
            instruction_dict = {"opcode": operation.opcode.value}
            if operation.arg is not None:
                if operation.arg.type == ArgType.PROGRAM_ADDRESS:
                    instruction_dict["arg"] = {
                        "value": operation.arg.value + instructions_offset,
                        "type": "ADDRESS"
                    }
                elif operation.arg.type == ArgType.DATA_ADDRESS:
                    instruction_dict["arg"] = {
                        "value": operation.arg.value,
                        "type": "ADDRESS"
                    }
                elif operation.arg.type == ArgType.DIRECT:
                    instruction_dict["arg"] = {
                        "value": operation.arg.value,
                        "type": "DIRECT"
                    }
                elif operation.arg.type == ArgType.INDIRECT_DATA_ADDRESS:
                    instruction_dict["arg"] = {
                        "value": operation.arg.value,
                        "type": "INDIRECT_ADDRESS"
                    }
            memory.append(instruction_dict)

        return json.dumps(
            {
                "start_idx": instructions_offset,
                "memory": memory,
            },
            indent=4)


def exec_binop(op: str, second_expr_result_addr: int, program: Program) -> None:
    if op == "+":
        program.operations.append(Operation(Opcode.ADD, Arg(second_expr_result_addr, ArgType.DATA_ADDRESS)))
    elif op == "=":
        program.operations.append(Operation(Opcode.EQ, Arg(second_expr_result_addr, ArgType.DATA_ADDRESS)))
    else:
        raise RuntimeError("Unknown binop")


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
        raise RuntimeError("Expected close bracket")


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
        result.load_int(int(tokens[idx].value))
        return get_expr_end_idx(tokens, idx + 1, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.BINOP:
        first_expr_end_idx: int = translate_expression(tokens, idx + 1, result)
        first_exp_res_addr = result.alloc_variable()
        result.operations.append(Operation(Opcode.ST, Arg(first_exp_res_addr, ArgType.DATA_ADDRESS)))
        second_expr_end_idx: int = translate_expression(tokens, first_expr_end_idx, result)
        exec_binop(tokens[idx].value, first_exp_res_addr, result)
        result.remove_intermediate_variable(first_exp_res_addr)
        return get_expr_end_idx(tokens, second_expr_end_idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.IF:
        condition_end_idx: int = translate_expression(tokens, idx + 1, result)
        je_idx: int = len(result.operations)
        result.operations.append(Operation(Opcode.JZ, None))
        true_branch_end_idx: int = translate_expression(tokens, condition_end_idx, result)
        jmp_idx: int = len(result.operations)
        result.operations.append(Operation(Opcode.JMP, None))
        false_branch_memory_idx: int = len(result.operations)
        false_branch_end_idx: int = translate_expression(tokens, true_branch_end_idx, result)
        result.operations[je_idx].arg = Arg(false_branch_memory_idx, ArgType.PROGRAM_ADDRESS)
        result.operations[jmp_idx].arg = Arg(len(result.operations), ArgType.PROGRAM_ADDRESS)
        return get_expr_end_idx(tokens, false_branch_end_idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.SETQ:
        if tokens[idx + 1].token_type != Token.Type.IDENTIFIER:
            raise RuntimeError(EXPECTED_IDENTIFIER)
        expr_end_idx: int = translate_expression(tokens, idx + 2, result)
        var_idx: int = result.alloc_variable(tokens[idx + 1].value)
        result.operations.append(Operation(Opcode.ST, Arg(var_idx, ArgType.DATA_ADDRESS)))
        return get_expr_end_idx(tokens, expr_end_idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.IDENTIFIER:
        result.operations.append(
            Operation(Opcode.LD, Arg(result.get_variable_address(tokens[idx].value), ArgType.DATA_ADDRESS)))
        return get_expr_end_idx(tokens, idx + 1, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.PROGN:
        idx += 1
        while tokens[idx].token_type == Token.Type.OPEN_BRACKET:
            idx = translate_expression(tokens, idx, result)
        return get_expr_end_idx(tokens, idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.STRING:
        string_addr = result.alloc_string(tokens[idx].value)
        result.load_string(string_addr)
        return get_expr_end_idx(tokens, idx + 1, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.PRINT:
        idx = translate_expression(tokens, idx + 1, result)
        if result.accum_value_type == AccumValueType.INT:
            result.operations.append(Operation(Opcode.OUT, None))
        else:
            str_len_addr_idx = result.alloc_variable()
            result.operations.append(Operation(Opcode.ST, Arg(str_len_addr_idx, ArgType.DATA_ADDRESS)))
            result.operations.append(Operation(Opcode.LD, Arg(str_len_addr_idx, ArgType.INDIRECT_DATA_ADDRESS)))

            str_len_idx = result.alloc_variable()
            result.operations.append(Operation(Opcode.ST, Arg(str_len_idx, ArgType.DATA_ADDRESS)))
            current_char_n_idx = result.alloc_variable()
            result.operations.append(Operation(Opcode.LD, Arg(0, ArgType.DIRECT)))
            result.operations.append(Operation(Opcode.ST, Arg(current_char_n_idx, ArgType.DATA_ADDRESS)))

            cycle_start_idx = len(result.operations)
            result.operations.append(Operation(Opcode.LD, Arg(current_char_n_idx, ArgType.DATA_ADDRESS)))
            result.operations.append(Operation(Opcode.EQ, Arg(str_len_idx, ArgType.DATA_ADDRESS)))
            result.operations.append(Operation(Opcode.LD, Arg(current_char_n_idx, ArgType.DATA_ADDRESS)))
            je_idx = len(result.operations)
            result.operations.append(Operation(Opcode.JNZ, None))
            result.operations.append(Operation(Opcode.ADD, Arg(1, ArgType.DIRECT)))
            result.operations.append(Operation(Opcode.ADD, Arg(str_len_addr_idx, ArgType.DATA_ADDRESS)))
            next_char_addr_idx = result.alloc_variable()
            result.operations.append(Operation(Opcode.ST, Arg(next_char_addr_idx, ArgType.DATA_ADDRESS)))
            result.operations.append(Operation(Opcode.LD, Arg(next_char_addr_idx, ArgType.INDIRECT_DATA_ADDRESS)))
            result.operations.append(Operation(Opcode.OUT, None))
            result.operations.append(Operation(Opcode.LD, Arg(current_char_n_idx, ArgType.DATA_ADDRESS)))
            result.operations.append(Operation(Opcode.ADD, Arg(1, ArgType.DIRECT)))
            result.operations.append(Operation(Opcode.ST, Arg(current_char_n_idx, ArgType.DATA_ADDRESS)))
            result.operations.append(Operation(Opcode.JMP, Arg(cycle_start_idx, ArgType.PROGRAM_ADDRESS)))
            result.operations[je_idx].arg = Arg(len(result.operations), ArgType.PROGRAM_ADDRESS)

            result.remove_intermediate_variable(str_len_idx)
            result.remove_intermediate_variable(current_char_n_idx)
            result.remove_intermediate_variable(next_char_addr_idx)
            result.remove_intermediate_variable(str_len_addr_idx)
        return get_expr_end_idx(tokens, idx, started_with_open_bracket)


def translate_program(tokens: list[Token], result: Program) -> None:
    translate_expression(tokens, 0, result)
    result.operations.append(Operation(Opcode.HLT, None))
