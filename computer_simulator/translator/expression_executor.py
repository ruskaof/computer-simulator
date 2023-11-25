import json
from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from computer_simulator.translator import Token


class Opcode(Enum):
    # M - memory address or direct value
    # AC - accumulator
    LD: int = auto()  # M -> AC
    ST: int = auto()  # AC -> M
    ADD: int = auto()  # AC + M -> AC
    SUB: int = auto()  # AC - M -> AC
    MUL: int = auto()  # AC * M -> AC
    DIV: int = auto()  # AC / M -> AC
    EQ: int = auto()  # AC == M -> AC
    JE: int = auto()  # if AC == 0 then PC = M
    JMP: int = auto()
    POP: int = auto()
    PUSH: int = auto()
    CALL: int = auto()
    RET: int = auto()


class Value:
    pass


@dataclass
class IntValue(Value):
    value: int


@dataclass
class StringValue(Value):
    memory_address: int


@dataclass
class Operation:
    class AddrType(Enum):
        DATA: int = auto()
        PROGRAM: int = auto()

    opcode: Opcode
    arg: Optional[Value | int]
    addr_type: Optional[AddrType]


class MemoryValue(ABC):
    pass


@dataclass
class IntMemoryValue(MemoryValue):
    value: int


@dataclass
class StringHeaderMemoryValue(MemoryValue):
    length: int


@dataclass
class StringCharMemoryValue(MemoryValue):
    char: str


class Program:
    _runtime_memory: list[MemoryValue | None] = []
    _max_memory_len: int = 0
    stack_variables: list[str] = []
    operations: list[Operation] = []
    runtime_acc_value: Value | int = 0
    runtime_variables: dict[str, int] = {}
    functions: dict[str, int] = {}

    def load_int(self, value: int) -> None:
        self.runtime_acc_value = IntValue(value)
        self.operations.append(Operation(Opcode.LD, self.runtime_acc_value, None))

    def get_last_operation_address(self) -> int:
        return int(len(self.operations) - 1)

    def append_empty_memory(self) -> int:
        self._runtime_memory.append(None)
        self._max_memory_len = max(self._max_memory_len, len(self._runtime_memory))
        return len(self._runtime_memory) - 1

    def pop_memory(self) -> None:
        self._runtime_memory.pop()

    def to_machine_code(self) -> str:
        ops = [
            {
                "opcode": op.opcode.name,
                "value_arg": op.arg.value if isinstance(op.arg, IntValue) else op.arg.memory_address if isinstance(
                    op.arg, StringValue) else op.arg,
            }
            for op in self.operations if op.arg is not None
        ]
        return json.dumps({"memory": [], "operations": ops}, indent=4)


def exec_binop(op: str, second_expr_result_addr: int, program: Program) -> None:
    if op == "+":
        program.operations.append(Operation(Opcode.ADD, second_expr_result_addr, Operation.AddrType.DATA))
    elif op == "-":
        program.operations.append(Operation(Opcode.SUB, second_expr_result_addr, Operation.AddrType.DATA))
    elif op == "*":
        program.operations.append(Operation(Opcode.MUL, second_expr_result_addr, Operation.AddrType.DATA))
    elif op == "/":
        program.operations.append(Operation(Opcode.DIV, second_expr_result_addr, Operation.AddrType.DATA))
    elif op == "=":
        program.operations.append(Operation(Opcode.EQ, second_expr_result_addr, Operation.AddrType.DATA))
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
    )


def get_expr_end_idx(tokens: list[Token], idx: int, started_with_open_bracket: bool) -> int:
    if tokens[idx].token_type == Token.Type.CLOSE_BRACKET and started_with_open_bracket:
        return idx + 1
    elif tokens[idx].token_type != Token.Type.CLOSE_BRACKET and not started_with_open_bracket:
        return idx
    else:
        raise RuntimeError("Expected close bracket")


def execute_expression(tokens: list[Token], idx: int, result: Program) -> int:
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
        first_expr_end_idx: int = execute_expression(tokens, idx + 1, result)
        first_exp_res_addr = result.append_empty_memory()
        result.operations.append(Operation(Opcode.ST, first_exp_res_addr, Operation.AddrType.DATA))
        second_expr_end_idx: int = execute_expression(tokens, first_expr_end_idx, result)
        exec_binop(tokens[idx].value, first_exp_res_addr, result)
        result.pop_memory()
        return get_expr_end_idx(tokens, second_expr_end_idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.IF:
        condition_end_idx: int = execute_expression(tokens, idx + 1, result)
        result.operations.append(Operation(Opcode.JE, None, Operation.AddrType.PROGRAM))
        je_addr: int = result.get_last_operation_address()
        true_branch_end_idx: int = execute_expression(tokens, condition_end_idx, result)
        result.operations.append(Operation(Opcode.JMP, None, Operation.AddrType.PROGRAM))
        jmp_addr: int = result.get_last_operation_address()
        false_branch_memory_addr: int = result.get_last_operation_address() + 1
        false_branch_end_idx: int = execute_expression(tokens, true_branch_end_idx, result)
        result.operations[je_addr].arg = false_branch_memory_addr
        result.operations[jmp_addr].arg = result.get_last_operation_address() + 1
        return get_expr_end_idx(tokens, false_branch_end_idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.SETQ:
        if tokens[idx + 1].token_type != Token.Type.IDENTIFIER:
            raise RuntimeError("Expected identifier")
        expr_end_idx: int = execute_expression(tokens, idx + 2, result)
        result.operations.append(Operation(Opcode.ST, result.runtime_variables[tokens[idx + 1].value], Operation.AddrType.DATA))
        return get_expr_end_idx(tokens, expr_end_idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.IDENTIFIER:
        result.operations.append(Operation(Opcode.LD, result.runtime_variables[tokens[idx].value], Operation.AddrType.DATA))
        return get_expr_end_idx(tokens, idx + 1, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.DEFUN:
        if tokens[idx + 1].token_type != Token.Type.IDENTIFIER:
            raise RuntimeError("Expected identifier")
        function_name: str = tokens[idx + 1].value
        if tokens[idx + 2].token_type != Token.Type.OPEN_BRACKET:
            raise RuntimeError("Expected open bracket")
        function_args: list[str] = []
        function_args_end_idx: int = idx + 3
        result.functions[function_name] = result.get_last_operation_address() + 1
        while tokens[function_args_end_idx].token_type != Token.Type.CLOSE_BRACKET:
            if tokens[function_args_end_idx].token_type != Token.Type.IDENTIFIER:
                raise RuntimeError("Expected identifier")
            function_args.append(tokens[function_args_end_idx].value)
            function_args_end_idx += 1

        result.stack_variables += function_args
        function_body_end_idx: int = execute_expression(tokens, function_args_end_idx + 1, result)
        for _ in function_args:
            result.stack_variables.pop()
            result.operations.append(Operation(Opcode.POP, None, None))

        result.operations.append(Operation(Opcode.RET, None, None))
        return get_expr_end_idx(tokens, function_body_end_idx, started_with_open_bracket)



