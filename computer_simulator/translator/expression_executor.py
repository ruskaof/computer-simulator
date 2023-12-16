import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from computer_simulator.isa import Arg as CompiledArg
from computer_simulator.isa import ArgType as CompiledArgType
from computer_simulator.isa import Opcode
from computer_simulator.translator import Token

EXPECTED_IDENTIFIER = "Expected identifier"
STATIC_MEMORY_SIZE = 512
DEFAULT_WORD = 0
NUMBER_OFFSET_IN_UTF8 = 48
STRING_ALLOC_SIZE = 32
SERVICE_VARIABLE_ADDRESS = 1


class ArgType(Enum):
    DIRECT: str = "DIRECT"
    DATA_ADDRESS: str = "DATA_ADDRESS"
    PROGRAM_ADDRESS: str = "PROGRAM_ADDRESS"
    INDIRECT: str = "INDIRECT"
    STACK_OFFSET: str = "STACK_OFFSET"


@dataclass
class Arg:
    value: Optional[int]
    type: ArgType

    def get_arg_for_compilation(self) -> CompiledArg:
        if self.type == ArgType.DIRECT:
            return CompiledArg(self.value, CompiledArgType.DIRECT)
        elif self.type == ArgType.DATA_ADDRESS:
            return CompiledArg(self.value, CompiledArgType.ADDRESS)
        elif self.type == ArgType.PROGRAM_ADDRESS:
            return CompiledArg(self.value + STATIC_MEMORY_SIZE, CompiledArgType.ADDRESS)
        elif self.type == ArgType.STACK_OFFSET:
            return CompiledArg(self.value, CompiledArgType.STACK_OFFSET)
        elif self.type == ArgType.INDIRECT:
            return CompiledArg(self.value, CompiledArgType.INDIRECT)
        else:
            raise RuntimeError("Unexpected arg type")


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
        self.memory: list[int] = [0, 0]  # jmp, service data
        self.operations: list[Operation] = []
        self.current_stack: list[StackValue] = []

    def load(self, value: int) -> None:
        self.operations.append(Operation(Opcode.LD, Arg(value, ArgType.DIRECT)))

    # allocates variable on top of stack
    def push_var_to_stack(self, name: Optional[str] = None) -> None:
        self.operations.append(Operation(Opcode.PUSH, None))
        self.current_stack.append(StackValue(len(self.memory), StackValue.Type.INT, name))

    def pop_var_from_stack(self) -> None:
        self.operations.append(Operation(Opcode.POP, None))
        self.current_stack.pop()

    def get_last_operation_index(self) -> int:
        return int(len(self.operations) - 1)

    def alloc_string(self, value: str) -> int:
        self.memory.append(len(value))
        self.operations.append(Operation(Opcode.LD, Arg(len(value), ArgType.DIRECT)))
        self.operations.append(Operation(Opcode.ST, Arg(len(self.memory) - 1, ArgType.DATA_ADDRESS)))
        for char in value:
            self.memory.append(ord(char))
            self.operations.append(Operation(Opcode.LD, Arg(ord(char), ArgType.DIRECT)))
            self.operations.append(Operation(Opcode.ST, Arg(len(self.memory) - 1, ArgType.DATA_ADDRESS)))
        return len(self.memory) - len(value) - 1

    def alloc_string_of_size(self, size: int) -> int:
        self.memory.append(size)
        self.operations.append(Operation(Opcode.LD, Arg(size, ArgType.DIRECT)))
        self.operations.append(Operation(Opcode.ST, Arg(len(self.memory) - 1, ArgType.DATA_ADDRESS)))
        for _ in range(size):
            self.memory.append(0)
        return len(self.memory) - size - 1

    def get_var_sp_offset(self, name: str) -> int:
        for i in range(len(self.current_stack) - 1, -1, -1):
            if self.current_stack[i].name == name:
                result = len(self.current_stack) - i
                return result

    def to_machine_code(self) -> str:
        memory = []

        for i in range(len(self.operations)):
            address = i + STATIC_MEMORY_SIZE if i > 0 else i
            if self.operations[i].arg is None:
                memory.append(
                    {
                        "opcode": self.operations[i].opcode.value,
                        "address": address,
                    }
                )
            else:
                arg = self.operations[i].arg.get_arg_for_compilation()

                memory.append(
                    {
                        "opcode": self.operations[i].opcode.value,
                        "arg": {
                            "value": arg.value,
                            "type": arg.type.value,
                        },
                        "address": address,
                    },
                )

        return json.dumps({"memory": memory}, indent=4)


def exec_binop(op: str, program: Program) -> None:
    if op == "+":
        program.operations.append(Operation(Opcode.ADD, Arg(1, ArgType.STACK_OFFSET)))
    elif op == "=":
        program.operations.append(Operation(Opcode.EQ, Arg(1, ArgType.STACK_OFFSET)))
    elif op == "%":
        program.operations.append(Operation(Opcode.MOD, Arg(1, ArgType.STACK_OFFSET)))
    elif op == "/":
        program.operations.append(Operation(Opcode.DIV, Arg(1, ArgType.STACK_OFFSET)))
    elif op == "<":
        program.operations.append(Operation(Opcode.LT, Arg(1, ArgType.STACK_OFFSET)))
    elif op == ">":
        program.operations.append(Operation(Opcode.GT, Arg(1, ArgType.STACK_OFFSET)))
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

        varname: str = tokens[idx + 1].value
        var_sp_offset: int = result.get_var_sp_offset(varname)
        if var_sp_offset is None:
            result.push_var_to_stack(varname)
            var_sp_offset = result.get_var_sp_offset(varname)

        result.operations.append(Operation(Opcode.ST, Arg(var_sp_offset, ArgType.STACK_OFFSET)))
        return get_expr_end_idx(tokens, expr_end_idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.IDENTIFIER:
        result.operations.append(
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
    elif tokens[idx].token_type == Token.Type.PRINT_CHAR:
        idx = translate_expression(tokens, idx + 1, result)
        result.operations.append(Operation(Opcode.OUT, None))
        return get_expr_end_idx(tokens, idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.PRINT_STRING:
        idx = translate_expression(tokens, idx + 1, result)

        result.push_var_to_stack("#str_p")
        # save string pointer
        result.operations.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#str_p"), ArgType.STACK_OFFSET)))

        # save string size:
        # load string size to ac
        result.operations.append(Operation(Opcode.ST, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.DATA_ADDRESS)))
        result.operations.append(Operation(Opcode.LD, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.INDIRECT)))

        # store string size
        result.push_var_to_stack("#str_size")
        result.operations.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#str_size"), ArgType.STACK_OFFSET)))

        # init index
        result.push_var_to_stack("#i")
        result.operations.append(Operation(Opcode.LD, Arg(0, ArgType.DIRECT)))
        result.operations.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))

        loop_start_idx: int = len(result.operations)
        # compare index with string size:
        # load index
        result.operations.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))
        result.operations.append(Operation(Opcode.EQ, Arg(result.get_var_sp_offset("#str_size"), ArgType.STACK_OFFSET)))

        jnz_idx: int = len(result.operations)
        # jump if index == string size
        result.operations.append(Operation(Opcode.JNZ, None))

        # load string pointer
        result.operations.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#str_p"), ArgType.STACK_OFFSET)))
        result.operations.append(Operation(Opcode.ADD, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))

        result.operations.append(Operation(Opcode.ADD, Arg(1, ArgType.DIRECT)))

        # load char
        result.operations.append(Operation(Opcode.ST, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.DATA_ADDRESS)))
        result.operations.append(Operation(Opcode.LD, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.INDIRECT)))
        result.operations.append(Operation(Opcode.OUT, None))

        # increment index
        result.operations.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))
        result.operations.append(Operation(Opcode.ADD, Arg(1, ArgType.DIRECT)))
        result.operations.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))

        # jump to compare index with string size
        result.operations.append(Operation(Opcode.JMP, Arg(loop_start_idx, ArgType.PROGRAM_ADDRESS)))
        result.operations[jnz_idx].arg = Arg(len(result.operations), ArgType.PROGRAM_ADDRESS)

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
        result.operations.append(Operation(Opcode.LD, Arg(string_addr, ArgType.DIRECT)))

        # save string pointer
        result.push_var_to_stack("#str_p")

        # index
        result.operations.append(Operation(Opcode.LD, Arg(1, ArgType.DIRECT)))
        result.push_var_to_stack("#i")

        # char
        result.push_var_to_stack("#char")

        # cycle start
        cycle_start_idx = len(result.operations)

        # read char
        result.operations.append(Operation(Opcode.IN, None))
        result.operations.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#char"), ArgType.STACK_OFFSET)))

        # if char is 0, then break
        result.operations.append(Operation(Opcode.EQ, Arg(0, ArgType.DIRECT)))
        jz_idx = len(result.operations)
        result.operations.append(Operation(Opcode.JNZ, None))

        # save char by index
        result.operations.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#str_p"), ArgType.STACK_OFFSET)))
        result.operations.append(Operation(Opcode.ADD, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))
        result.operations.append(Operation(Opcode.ST, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.DATA_ADDRESS)))
        result.operations.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#char"), ArgType.STACK_OFFSET)))
        result.operations.append(Operation(Opcode.ST, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.INDIRECT)))

        # increment index
        result.operations.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))
        result.operations.append(Operation(Opcode.ADD, Arg(1, ArgType.DIRECT)))
        result.operations.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))

        # jump to cycle start
        result.operations.append(Operation(Opcode.JMP, Arg(cycle_start_idx, ArgType.PROGRAM_ADDRESS)))
        result.operations[jz_idx].arg = Arg(len(result.operations), ArgType.PROGRAM_ADDRESS)

        # save string size
        result.operations.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#str_p"), ArgType.STACK_OFFSET)))
        result.operations.append(Operation(Opcode.ST, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.DATA_ADDRESS)))
        result.operations.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))
        result.operations.append(Operation(Opcode.SUB, Arg(1, ArgType.DIRECT)))
        result.operations.append(Operation(Opcode.ST, Arg(SERVICE_VARIABLE_ADDRESS, ArgType.INDIRECT)))

        # save string pointer to variable
        result.operations.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#str_p"), ArgType.STACK_OFFSET)))
        result.operations.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset(varname), ArgType.STACK_OFFSET)))

        result.pop_var_from_stack()  # i
        result.pop_var_from_stack()  # char
        result.pop_var_from_stack()  # str_p

        result.operations.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset(varname), ArgType.STACK_OFFSET)))

        return get_expr_end_idx(tokens, idx + 2, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.WHILE:
        loop_start_idx = len(result.operations)
        condition_end_idx = translate_expression(tokens, idx + 1, result)
        jz_idx = len(result.operations)
        result.operations.append(Operation(Opcode.JZ, None))
        body_end_idx = translate_expression(tokens, condition_end_idx, result)
        result.operations.append(Operation(Opcode.JMP, Arg(loop_start_idx, ArgType.PROGRAM_ADDRESS)))
        result.operations[jz_idx].arg = Arg(len(result.operations), ArgType.PROGRAM_ADDRESS)

        return get_expr_end_idx(tokens, body_end_idx, started_with_open_bracket)


def translate_program(tokens: list[Token], result: Program) -> None:
    result.operations.append(Operation(Opcode.JMP, Arg(1, ArgType.PROGRAM_ADDRESS)))
    translate_expression(tokens, 0, result)
    result.operations.append(Operation(Opcode.HLT, None))
