import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import cast, Optional

from computer_simulator.translator import Token, MemoryAddress


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


class Value:
    pass


@dataclass
class IntValue(Value):
    value: int


@dataclass
class StringValue(Value):
    memory_address: MemoryAddress


@dataclass
class Operation:
    opcode: Opcode
    arg: Optional[Value | MemoryAddress]


class Heap:
    memory: list[int] = []
    _variables: dict[str, MemoryAddress] = {}

    def allocate_string(self, value: str) -> MemoryAddress:
        self.memory.append(len(value) + 1)
        for char in value:
            self.memory.append(ord(char))

        return MemoryAddress(len(self.memory) - 1)

    def init_variable(self, name: str) -> MemoryAddress:
        self._variables[name] = MemoryAddress(len(self.memory))
        self.memory.append(0)
        return self._variables[name]

    def save_value(self, value: Optional[Value]) -> MemoryAddress:
        if isinstance(value, IntValue):
            self.memory.append(value.value)
            return MemoryAddress(len(self.memory) - 1)
        elif isinstance(value, StringValue):
            self.memory.append(value.memory_address)
            return MemoryAddress(len(self.memory) - 1)
        elif value is None:
            self.memory.append(0)
            return MemoryAddress(len(self.memory) - 1)

    def get_variable(self, name: str) -> MemoryAddress:
        return self._variables[name]


class Program:
    _heap: Heap = Heap()
    _operations: list[Operation] = []

    def load_variable(self, name: str) -> None:
        self._operations.append(Operation(Opcode.LD, StringValue(self._heap.get_variable(name))))

    def load_int(self, value: int) -> None:
        self._operations.append(Operation(Opcode.LD, IntValue(value)))

    def load_string(self, value: str) -> None:
        self._operations.append(Operation(Opcode.LD, StringValue(self._heap.allocate_string(value))))

    def save_intermediate(self) -> MemoryAddress:
        address: MemoryAddress = self._heap.save_value(None)
        self._operations.append(Operation(Opcode.ST, address))
        return self._heap.save_value(None)

    def exec_binop(self, op: str, second_expr_result_addr: MemoryAddress) -> None:
        if op == "+":
            self._operations.append(Operation(Opcode.ADD, second_expr_result_addr))
        elif op == "-":
            self._operations.append(Operation(Opcode.SUB, second_expr_result_addr))
        elif op == "*":
            self._operations.append(Operation(Opcode.MUL, second_expr_result_addr))
        elif op == "/":
            self._operations.append(Operation(Opcode.DIV, second_expr_result_addr))
        elif op == "=":
            self._operations.append(Operation(Opcode.EQ, second_expr_result_addr))
        else:
            raise RuntimeError("Unknown binop")

    def to_machine_code(self) -> str:
        memory = []
        ops = []
        for op in self._operations:
            res: dict = {"opcode": op.opcode.name}
            if op.arg is not None:
                if isinstance(op.arg, IntValue):
                    res["value_arg"] = op.arg.value
                elif isinstance(op.arg, StringValue):
                    res["value_arg"] = op.arg.memory_address
                elif isinstance(op.arg, int):
                    res["memory_arg"] = op.arg
                else:
                    raise RuntimeError("Unknown arg type")
            ops.append(res)
        for mem in self._heap.memory:
            memory.append(mem)
        return json.dumps({"memory": memory, "operations": ops}, indent=4)


def execute_expression(tokens: list[Token], idx: int, result: Program) -> int:
    if idx >= len(tokens):
        return idx
    elif tokens[idx].token_type != Token.Type.OPEN_BRACKET:
        raise RuntimeError("Expected open bracket")
    elif tokens[idx + 1].token_type == Token.Type.IDENTIFIER:
        result.load_variable(tokens[idx + 1].value)
        return idx + 3
    elif tokens[idx + 1].token_type == Token.Type.INT:
        result.load_int(int(tokens[idx + 1].value))
        return idx + 3
    elif tokens[idx + 1].token_type == Token.Type.STRING:
        result.load_string(tokens[idx + 1].value)
        return idx + 3
    elif tokens[idx + 1].token_type == Token.Type.BINOP:
        first_expr_end_idx: int = execute_expression(tokens, idx + 2, result)
        first_expr_result_addr: MemoryAddress = result.save_intermediate()
        second_expr_end_idx: int = execute_expression(tokens, first_expr_end_idx, result)
        result.exec_binop(tokens[idx + 1].value, first_expr_result_addr)
        return second_expr_end_idx
    else:
        raise RuntimeError("Unknown no value token type")
