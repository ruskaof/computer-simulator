import logging
import sys
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
import json
from typing import Optional

from computer_simulator.isa import Operation, Opcode, Arg, ArgType

WORD_SIZE: int = 64
WORD_MAX_VALUE: int = 2 ** WORD_SIZE


@dataclass
class Port(Enum):
    IN = "IN"
    OUT = "OUT"


@dataclass
class BinaryProgram:
    memory: list[Operation | int]
    start_idx: int


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_json_file(file_path: str) -> dict:
    return json.loads(read_file(file_path))


def read_input_as_pascal_string(file_path: str) -> list[int]:
    with open(file_path, "r", encoding="utf-8") as file:
        file_str = file.read()
        return [len(file_str)] + [ord(c) for c in file_str]


def read_program(exe: str) -> BinaryProgram:
    json_exe = read_json_file(exe)
    memory = []
    for word in json_exe["memory"]:
        if isinstance(word, int):
            memory.append(word)
        else:
            arg = None
            if "arg" in word:
                arg = Arg(word["arg"]["value"], ArgType(word["arg"]["type"]))
            memory.append(Operation(Opcode(word["opcode"]), arg))
    start_idx = json_exe["start_idx"]
    return BinaryProgram(memory, start_idx)


class IpSelSignal(Enum):
    INC = 0
    DR = 1


class SpSelSignal(Enum):
    INC = 0
    DEC = 1


class ArSelSignal(Enum):
    DR = 0
    SP = 1
    IP = 2


class AcSelSignal(Enum):
    IN = 0
    ALU = 1
    DR = 2


class DrSelSignal(Enum):
    INSTRUCTION_DECODER = 0
    MEMORY = 1


class AluOp(Enum):
    ADD = 0
    EQ = 1


class Alu:
    value: int = 0
    flag_z: bool = True

    def perform(self, op: AluOp, left: int, right: int) -> None:
        if op == AluOp.ADD:
            self.value = (left + right) % WORD_MAX_VALUE
        elif op == AluOp.EQ:
            self.value = 1 if left == right else 0
        else:
            raise RuntimeError(f"Unknown ALU operation: {op}")
        self.set_flags()

    def set_flags(self) -> None:
        self.flag_z = self.value == 0


class DataPath:
    memory: list[Operation | int]
    ports: dict[str, list[int]]
    alu: Alu = Alu()
    ip: int = 0  # instruction pointer
    dr: int = 0  # data register
    sp: int = 0  # stack pointer
    ar: int = 0  # address register
    ac: int = 0  # accumulator

    def __init__(self, memory: list[Operation | int], ports: dict[str, list[int]], start_idx: int) -> None:
        self.memory = memory
        self.ports = ports
        self.ip = start_idx

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

    def latch_ac(self, signal: AcSelSignal, alu_op: Optional[AluOp] = None) -> None:
        if signal == AcSelSignal.IN:
            self.ac = self.memory[self.ar]
        elif signal == AcSelSignal.ALU:
            self.alu.perform(alu_op, self.ac, self.dr)
            self.ac = self.alu.value
        elif signal == AcSelSignal.DR:
            self.ac = self.dr
        else:
            self._rase_for_unknown_signal(signal)

    def latch_dr(self, signal: DrSelSignal) -> Optional[Operation]:
        if signal == DrSelSignal.INSTRUCTION_DECODER:
            cell = self.memory[self.ip]
            if cell.arg is not None:
                self.dr = self.memory[self.ip].arg.value
            return self.memory[self.ip]
        elif signal == DrSelSignal.MEMORY:
            self.dr = self.memory[self.ar]
            return None
        else:
            self._rase_for_unknown_signal(signal)

    def latch_out(self) -> None:
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

    def next(self) -> 'Stage':
        return Stage((self.value + 1) % Stage.__len__())


class ControlUnit:
    data_path: DataPath
    tick_n: int = 0
    stage: Stage = Stage.INSTRUCTION_FETCH
    decoded_instruction: Optional[Operation] = None
    halted: bool = False

    def __init__(self, data_path: DataPath):
        self.data_path = data_path

    def tick(self) -> None:
        if self.stage == Stage.INSTRUCTION_FETCH:
            self.data_path.latch_ar(ArSelSignal.IP)
            self.decoded_instruction = self.data_path.latch_dr(DrSelSignal.INSTRUCTION_DECODER)
            self.stage = self.stage.next()
            self.tick_n += 1
        elif self.stage == Stage.ADDRESS_FETCH:
            if self.decoded_instruction is None:
                raise RuntimeError("Instruction is not decoded")
            if self.decoded_instruction.arg is not None and self.decoded_instruction.arg.type == ArgType.INDIRECT_ADDRESS:
                self.data_path.latch_ar(ArSelSignal.DR)
                self.data_path.latch_dr(DrSelSignal.INSTRUCTION_DECODER)
                self.decoded_instruction.arg.type = ArgType.ADDRESS
                self.decoded_instruction.arg.value = self.data_path.dr
                self.tick_n += 1
                self.stage = self.stage.next()
            else:
                self.stage = self.stage.next()
                self.tick()
        elif self.stage == Stage.OPERAND_FETCH:
            if self.decoded_instruction is None:
                raise RuntimeError("Instruction is not decoded")
            if self.decoded_instruction.arg is not None and self.decoded_instruction.arg.type == ArgType.ADDRESS:
                self.data_path.latch_ar(ArSelSignal.DR)
                self.data_path.latch_dr(DrSelSignal.INSTRUCTION_DECODER)
                self.tick_n += 1
                self.stage = self.stage.next()
            else:
                self.stage = self.stage.next()
                self.tick()
        elif self.stage == Stage.EXECUTE:
            if self.decoded_instruction is None:
                raise RuntimeError("Instruction is not decoded")
            if self.decoded_instruction.opcode == Opcode.LD:
                self.data_path.latch_ac(AcSelSignal.DR)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.ST:
                self.data_path.wr()
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.ADD:
                self.data_path.latch_ac(AcSelSignal.ALU, AluOp.ADD)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.EQ:
                self.data_path.latch_ac(AcSelSignal.ALU, AluOp.EQ)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.JE:
                if self.data_path.alu.flag_z:
                    self.data_path.latch_ip(IpSelSignal.DR)
                    self.tick_n += 1
                else:
                    self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.JMP:
                self.data_path.latch_ip(IpSelSignal.DR)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.POP:
                self.data_path.latch_sp(SpSelSignal.DEC)
                self.data_path.latch_ar(ArSelSignal.SP)
                self.data_path.latch_dr(DrSelSignal.MEMORY)
                self.data_path.latch_ac(AcSelSignal.DR)
                self.tick_n += 1
                self.stage = self.stage.next()
            elif self.decoded_instruction.opcode == Opcode.PUSH:
                self.data_path.latch_ar(ArSelSignal.SP)
                self.data_path.latch_sp(SpSelSignal.INC)
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
            else:
                raise RuntimeError(f"Unknown opcode: {self.decoded_instruction.opcode}")
            self.data_path.latch_ip(IpSelSignal.INC)

    def __repr__(self):
        return (f"TICK: {self.tick_n}, IP: {self.data_path.ip}, DR: {self.data_path.dr}, SP: {self.data_path.sp}, "
                f"AR: {self.data_path.ar}, AC: {self.data_path.ac}, ALU: {self.data_path.alu.value}, "
                f"Z: {self.data_path.alu.flag_z}, INSTR: {self.decoded_instruction}")


def simulation(program: BinaryProgram, limit: int, program_input: list[int]) -> list[int]:
    data_path = DataPath(program.memory, {Port.IN.name: program_input, Port.OUT.name: []}, program.start_idx)
    control_unit = ControlUnit(data_path)

    logging.debug("%s", control_unit)
    while not control_unit.halted and control_unit.tick_n < limit:
        control_unit.tick()
        logging.debug("%s", control_unit)

    return data_path.ports[Port.OUT.name]


def main(code: str, input_file: str) -> None:
    program: BinaryProgram = read_program(code)
    program_input: list[int] = read_input_as_pascal_string(input_file)

    result = simulation(program, limit=1000, program_input=program_input)

    print("".join(chr(c) for c in result))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    assert len(sys.argv) >= 3, f"Usage: python3 {Path(__file__).name} <code> <input>"
    main(sys.argv[1], sys.argv[2])
