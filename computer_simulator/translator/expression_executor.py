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


class ArgType(Enum):
    DIRECT: str = "DIRECT"
    DATA_ADDRESS: str = "DATA_ADDRESS"
    PROGRAM_ADDRESS: str = "PROGRAM_ADDRESS"
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
        self.memory: list[int] = [0]
        self.operations: list[Operation] = []
        self.current_stack: list[StackValue] = []

    def load(self, value: int) -> None:
        self.operations.append(Operation(Opcode.LD, Arg(value, ArgType.DIRECT)))

    # allocates variable on top of stack
    def alloc_variable(self, name: Optional[str] = None) -> None:
        self.operations.append(Operation(Opcode.PUSH, None))
        self.current_stack.append(StackValue(len(self.memory), StackValue.Type.INT, name))

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

    def get_var_sp_offset(self, name: str) -> int:
        for i in range(len(self.current_stack) - 1, -1, -1):
            if self.current_stack[i].name == name:
                return len(self.current_stack) - i

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
        result.operations.append(Operation(Opcode.ST, Arg(1, ArgType.STACK_OFFSET)))
        second_expr_end_idx: int = translate_expression(tokens, first_expr_end_idx, result)
        exec_binop(tokens[idx].value, result)
        result.current_stack.pop()
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
        result.alloc_variable(tokens[idx + 1].value)
        result.operations.append(Operation(Opcode.ST, Arg(1, ArgType.STACK_OFFSET)))
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
    elif tokens[idx].token_type == Token.Type.PRINT_INT:
        idx = translate_expression(tokens, idx + 1, result)
        result.operations.append(Operation(Opcode.ADD, Arg(NUMBER_OFFSET_IN_UTF8, ArgType.DIRECT)))
        result.operations.append(Operation(Opcode.OUT, None))
        return get_expr_end_idx(tokens, idx, started_with_open_bracket)
    elif tokens[idx].token_type == Token.Type.PRINT_STRING:
        idx = translate_expression(tokens, idx + 1, result)

        result.alloc_variable("#str_p")
        # save string pointer
        result.operations.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#str_p"), ArgType.STACK_OFFSET)))

        # save string size:
        # load string size to ac
        result.operations.append(Operation(Opcode.LD_BY_AC, None))
        # store string size
        result.alloc_variable("#str_size")
        result.operations.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#str_size"), ArgType.STACK_OFFSET)))

        # init index
        result.alloc_variable("#i")
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
        result.operations.append(Operation(Opcode.LD_BY_AC, None))
        result.operations.append(Operation(Opcode.OUT, None))

        # increment index
        result.operations.append(Operation(Opcode.LD, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))
        result.operations.append(Operation(Opcode.ADD, Arg(1, ArgType.DIRECT)))
        result.operations.append(Operation(Opcode.ST, Arg(result.get_var_sp_offset("#i"), ArgType.STACK_OFFSET)))

        # jump to compare index with string size
        result.operations.append(Operation(Opcode.JMP, Arg(loop_start_idx, ArgType.PROGRAM_ADDRESS)))
        result.operations[jnz_idx].arg = Arg(len(result.operations), ArgType.PROGRAM_ADDRESS)

        result.current_stack.pop()  # i
        result.current_stack.pop()  # str_size
        result.current_stack.pop()  # str_p
        return get_expr_end_idx(tokens, idx, started_with_open_bracket)


def translate_program(tokens: list[Token], result: Program) -> None:
    result.operations.append(Operation(Opcode.JMP, Arg(1, ArgType.PROGRAM_ADDRESS)))
    translate_expression(tokens, 0, result)
    result.operations.append(Operation(Opcode.HLT, None))
