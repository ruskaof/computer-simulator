from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from computer_simulator.isa import Arg, ArgType, Opcode, Operation

WORD_SIZE: int = 64
WORD_MAX_VALUE: int = 2**WORD_SIZE
MEMORY_SIZE: int = 2048


@dataclass
class Port(Enum):
    IN = "IN"
    OUT = "OUT"


@dataclass
class BinaryProgram:
    memory: list[Operation | int]


def read_file(file_path: str) -> str:
    with open(file_path, encoding="utf-8") as file:
        return file.read()


def read_json_file(file_path: str) -> dict:
    return json.loads(read_file(file_path))


def read_input(file_path: str) -> list[int]:
    with open(file_path, encoding="utf-8") as file:
        file_str = file.read()
        return [ord(c) for c in file_str]


def read_program(exe: str) -> BinaryProgram:
    json_exe = read_json_file(exe)
    memory = [0 for _ in range(MEMORY_SIZE)]
    for word in json_exe["memory"]:
        arg = None
        if "arg" in word:
            arg = Arg(word["arg"]["value"], ArgType(word["arg"]["type"]))
        comment: str | None = None
        if "comment" in word:
            comment = word["comment"]
        address: int = word["address"]
        memory[address] = Operation(Opcode(word["opcode"]), arg, comment)
    return BinaryProgram(memory)


class IpSelSignal(Enum):
    INC = 0
    DR = 1


class SpSelSignal(Enum):
    INC = 0
    DEC = 1


class ArSelSignal(Enum):
    AC = 0
    SP = 1
    IP = 2
    DR = 3


class AcSelSignal(Enum):
    IN = 0
    ALU = 1
    DR = 2
    IP = 3


class DrSelSignal(Enum):
    INSTRUCTION_DECODER = 0
    MEMORY = 1
    ALU = 2


class AluOp(Enum):
    ADD = 0
    SUB = 1
    EQ = 2
    GT = 3
    LT = 4
    MOD = 5
    DIV = 6
    MULT = 7

class Alu:
    def __init__(self):
        self.flag_z: bool = True

    def perform(self, op: AluOp, left: int, right: int) -> int:
        if op == AluOp.ADD:
            value = (left + right) % WORD_MAX_VALUE
            self.set_flags(value)
        elif op == AluOp.SUB:
            value = (left - right) % WORD_MAX_VALUE
            if value < 0:
                raise RuntimeError("ALU value is negative")
            self.set_flags(value)
        elif op == AluOp.EQ:
            value = 1 if left == right else 0
            self.set_flags(value)
        elif op == AluOp.GT:
            value = 1 if left > right else 0
            self.set_flags(value)
        elif op == AluOp.LT:
            value = 1 if left < right else 0
            self.set_flags(value)
        elif op == AluOp.MOD:
            value = left % right
            self.set_flags(value)
        elif op == AluOp.DIV:
            value = left // right
            self.set_flags(value)
        elif op == AluOp.MULT:
            value = left * right
            self.set_flags(value)
        else:
            raise RuntimeError(f"Unknown ALU operation: {op}")
        return value

    def set_flags(self, value) -> None:
        self.flag_z = value == 0


class DataPath:
    def __init__(self, memory: list[Operation | int], ports: dict[str, list[int]]) -> None:
        self.memory: list[Operation | int] = memory
        self.ports: dict[str, list[int]] = ports
        self.alu: Alu = Alu()
        self.ip: int = 0  # instruction pointer
        self.dr: int = 0  # data register
        self.sp: int = len(self.memory) # stack pointer
        self.ar: int = 0  # address register
        self.ac: int = 0  # accumulator

    def latch_ip(self, signal: IpSelSignal) -> None:
        if signal == IpSelSignal.INC:
            self.ip += 1
        elif signal == IpSelSignal.DR:
            self.ip = self.dr
        else:
            self._rase_for_unknown_signal(signal)

    def latch_sp(self, signal: SpSelSignal) -> None:
        if signal == SpSelSignal.INC:
            self.sp += 1
        elif signal == SpSelSignal.DEC:
            self.sp -= 1
        else:
            self._rase_for_unknown_signal(signal)

    def latch_ar(self, signal: ArSelSignal) -> None:
        if signal == ArSelSignal.DR:
            self.ar = self.dr
        elif signal == ArSelSignal.SP:
            self.ar = self.sp
        elif signal == ArSelSignal.IP:
            self.ar = self.ip
        else:
            self._rase_for_unknown_signal(signal)

    def latch_ac(self, signal: AcSelSignal, alu_op: AluOp | None = None) -> None:
        if signal == AcSelSignal.IN:
            if len(self.ports[Port.IN.name]) == 0:
                self.ac = 0
                logging.debug("IN: %s", self.ac)
            else:
                self.ac = self.ports[Port.IN.name].pop(0)
                logging.debug('IN: %s - "%s"', self.ac, chr(self.ac))
        elif signal == AcSelSignal.ALU:
            self.ac = self.alu.perform(alu_op, self.ac, self.dr)
        elif signal == AcSelSignal.DR:
            self.ac = self.dr
        elif signal == AcSelSignal.IP:
            self.ac = self.ip
        else:
            self._rase_for_unknown_signal(signal)

    def latch_dr(self, signal: DrSelSignal, alu_res: int | None = None) -> Operation | None:
        if signal == DrSelSignal.INSTRUCTION_DECODER:
            cell = self.memory[self.ip]
            if isinstance(cell, int):
                raise RuntimeError(f"Expected instruction, got {cell}")
            if cell.arg is not None:
                self.dr = self.memory[self.ip].arg.value
            return self.memory[self.ip]
        elif signal == DrSelSignal.MEMORY:
            self.dr = self.memory[self.ar]
            return None
        elif signal == DrSelSignal.ALU:
            self.dr = alu_res
            return None
        else:
            self._rase_for_unknown_signal(signal)
            return None

    def latch_out(self) -> None:
        logging.debug('OUT: %s - "%s"', self.ac, chr(self.ac))
        self.ports[Port.OUT.name].append(self.ac)

    def wr(self):
        self.memory[self.ar] = self.ac

    def _rase_for_unknown_signal(self, unknown_signal: any) -> None:
        raise RuntimeError(f"Unknown signal: {unknown_signal}")


class Stage(Enum):
    INSTRUCTION_FETCH = 0
    ADDRESS_FETCH = 1
    OPERAND_FETCH = 2
    EXECUTE = 3

    def next(self) -> Stage:
        return Stage((self.value + 1) % Stage.__len__())


NO_FETCH_OPERAND = [
    Opcode.JMP,
    Opcode.JZ,
    Opcode.JNZ,
    Opcode.ST,
    Opcode.PUSH,
    Opcode.POP,
    Opcode.CALL,
]


class ControlUnit:
    def __init__(self, data_path: DataPath):
        self.data_path: DataPath = data_path
        self.tick_n: int = 0
        self.stage: Stage = Stage.INSTRUCTION_FETCH
        self.decoded_instruction: Operation | None = None
        self.halted: bool = False
        self.executed_instruction_n: int = 0

    def tick(self) -> None:
        if self.stage == Stage.INSTRUCTION_FETCH:
            self.data_path.latch_ar(ArSelSignal.IP)
            self.decoded_instruction = self.data_path.latch_dr(DrSelSignal.INSTRUCTION_DECODER)
            self.stage = self.stage.next()
            self.tick_n += 1
        elif self.stage == Stage.ADDRESS_FETCH:
            if self.decoded_instruction is None:
                raise RuntimeError("Instruction is not decoded")
            elif self.decoded_instruction.arg is not None and self.decoded_instruction.arg.arg_type == ArgType.STACK_OFFSET:
                alu_res = self.data_path.alu.perform(AluOp.ADD, self.data_path.sp, self.data_path.dr)
                self.data_path.latch_dr(DrSelSignal.ALU, alu_res)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.arg is not None and self.decoded_instruction.arg.arg_type == ArgType.INDIRECT:
                self.data_path.latch_ar(ArSelSignal.DR)
                self.data_path.latch_dr(DrSelSignal.MEMORY)
                self.tick_n += 1
                self.stage = self.stage.next()
            else:
                self.stage = self.stage.next()
        elif self.stage == Stage.OPERAND_FETCH:
            if self.decoded_instruction is None:
                raise RuntimeError("Instruction is not decoded")
            if (
                self.decoded_instruction.arg is not None
                and self.decoded_instruction.opcode not in NO_FETCH_OPERAND
                and self.decoded_instruction.arg.arg_type in (ArgType.STACK_OFFSET, ArgType.ADDRESS, ArgType.INDIRECT)
            ):
                self.data_path.latch_ar(ArSelSignal.DR)
                self.data_path.latch_dr(DrSelSignal.MEMORY)
                self.tick_n += 1
                self.stage = self.stage.next()
            else:
                self.stage = self.stage.next()
                self.tick()
        elif self.stage == Stage.EXECUTE:
            should_inc_ip = True
            if self.decoded_instruction is None:
                raise RuntimeError("Instruction is not decoded")
            if self.decoded_instruction.opcode == Opcode.LD:
                self.data_path.latch_ac(AcSelSignal.DR)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.ST:
                self.data_path.latch_ar(ArSelSignal.DR)
                self.data_path.wr()
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.ADD:
                self.data_path.latch_ac(AcSelSignal.ALU, AluOp.ADD)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.SUB:
                self.data_path.latch_ac(AcSelSignal.ALU, AluOp.SUB)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.EQ:
                self.data_path.latch_ac(AcSelSignal.ALU, AluOp.EQ)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.JZ:
                if self.data_path.alu.flag_z:
                    self.data_path.latch_ip(IpSelSignal.DR)
                    self.tick_n += 1
                    should_inc_ip = False
                else:
                    self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.JNZ:
                if not self.data_path.alu.flag_z:
                    self.data_path.latch_ip(IpSelSignal.DR)
                    self.tick_n += 1
                    should_inc_ip = False
                else:
                    self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.JMP:
                self.data_path.latch_ip(IpSelSignal.DR)
                self.tick_n += 1
                should_inc_ip = False
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.POP:
                self.data_path.latch_ar(ArSelSignal.SP)
                self.data_path.latch_dr(DrSelSignal.MEMORY)
                self.data_path.latch_sp(SpSelSignal.INC)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.PUSH:
                self.data_path.latch_sp(SpSelSignal.DEC)
                self.data_path.latch_ar(ArSelSignal.SP)
                self.data_path.wr()
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.IN:
                self.data_path.latch_ac(AcSelSignal.IN)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.OUT:
                self.data_path.latch_out()
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.HLT:
                self.halted = True
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.LT:
                self.data_path.latch_ac(AcSelSignal.ALU, AluOp.LT)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.GT:
                self.data_path.latch_ac(AcSelSignal.ALU, AluOp.GT)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.CALL:
                self.data_path.latch_ac(AcSelSignal.IP)
                self.data_path.latch_sp(SpSelSignal.DEC)
                self.data_path.latch_ar(ArSelSignal.SP)
                self.data_path.latch_ip(IpSelSignal.DR)
                self.data_path.wr()
                self.tick_n += 1
                should_inc_ip = False
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.RET:
                self.data_path.latch_ar(ArSelSignal.SP)
                self.data_path.latch_dr(DrSelSignal.MEMORY)
                self.data_path.latch_ip(IpSelSignal.DR)
                self.data_path.latch_sp(SpSelSignal.INC)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.MOD:
                self.data_path.latch_ac(AcSelSignal.ALU, AluOp.MOD)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.DIV:
                self.data_path.latch_ac(AcSelSignal.ALU, AluOp.DIV)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.MUL:
                self.data_path.latch_ac(AcSelSignal.ALU, AluOp.MULT)
                self.tick_n += 1
                self.stage = self.stage.next()
            else:
                raise RuntimeError(f"Unknown opcode: {self.decoded_instruction.opcode}")
            if should_inc_ip:
                self.data_path.latch_ip(IpSelSignal.INC)
            self.executed_instruction_n += 1

    def __repr__(self):
        stack_str = ""
        for i in range(0, len(self.data_path.memory)):
            if self.data_path.sp + i < len(self.data_path.memory):
                stack_str += f"{self.data_path.memory[self.data_path.sp + i]} "
            else:
                break

        return (
            f"TICK: {self.tick_n}, IP: {self.data_path.ip}, DR: {self.data_path.dr}, "
            f"AR: {self.data_path.ar}, AC: {self.data_path.ac}, "
            f"Z: {self.data_path.alu.flag_z}, INSTR: {self.decoded_instruction}, SP: {self.data_path.sp}, "
            f"Stack: {stack_str}"
        )


def simulation(program: BinaryProgram, limit: int, program_input: list[int]) -> tuple[list[int], int, int]:
    """
    Simulate program execution
    :return: output, instructions_n, ticks_n
    """
    data_path: DataPath = DataPath(program.memory, {Port.IN.name: program_input, Port.OUT.name: []})
    control_unit: ControlUnit = ControlUnit(data_path)

    logging.debug("%s", control_unit)
    while not control_unit.halted and control_unit.tick_n < limit:
        control_unit.tick()
        logging.debug("%s", control_unit)

    return data_path.ports[Port.OUT.name], control_unit.executed_instruction_n, control_unit.tick_n


def main(code: str, input_file: str) -> None:
    program: BinaryProgram = read_program(code)
    program_input: list[int] = read_input(input_file)

    result = simulation(program, limit=10_000, program_input=program_input)

    print("".join([chr(c) for c in result[0]]))
    print(f"instructions_n: {result[1]} ticks: {result[2]}")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    assert len(sys.argv) >= 3, f"Usage: python3 {Path(__file__).name} <code> <input>"
    main(sys.argv[1], sys.argv[2])
