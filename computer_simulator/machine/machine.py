import sys
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
import json
from typing import Optional

from computer_simulator.isa import Operation


@dataclass
class BinaryProgram:
    memory: list[int]
    operations: list[Operation]
    start_idx: int
    ports_data: dict[int, list[int]]


def read_json_file(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.loads(file.read())


def read_program(exe: str, input: str) -> BinaryProgram:
    json_exe = read_json_file(exe)
    operations = [
        Operation(
            op["opcode"],
            op.get("value_arg", op["address_arg"]),
            Operation.ArgType.DIRECT if "value_arg" in op else Operation.ArgType.ADDRESS,
        )
        for op in json_exe["operations"]
    ]
    start_idx = json_exe["start_idx"]
    ports_data = {port["num"]: port["data"] for port in read_json_file(input)["ports"]}
    return BinaryProgram(json_exe["memory"], operations, start_idx, ports_data)



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
    MEM = 1
    DR = 2

class DrSelSignal(Enum):
    INSTRUCTION_DECODER = 0
    MEMORY = 1

class DataPath:
    memory: list[Operation | int]
    ports: dict[int, list[int]]
    ip: int = 0 # instruction pointer
    dr: int = 0 # data register
    sp: int = 0 # stack pointer
    ar: int = 0 # address register
    ac: int = 0 # accumulator

    def __init__(self, memory: list[Operation | int], ports: dict[int, list[int]]):
        self.memory = memory
        self.ports = ports

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

    def latch_ac(self, signal: AcSelSignal) -> None:
        if signal == AcSelSignal.IN:
            self.ac = self.memory[self.ar]
        elif signal == AcSelSignal.MEM:
            self.memory[self.ar] = self.ac
        elif signal == AcSelSignal.DR:
            self.ac = self.dr
        else:
            self._rase_for_unknown_signal(signal)

    def latch_dr(self, signal: DrSelSignal):
        if signal == DrSelSignal.INSTRUCTION_DECODER:
            self.dr = self.memory[self.ip].arg
        elif signal == DrSelSignal.MEMORY:
            self.dr = self.memory[self.ar]
        else:
            self._rase_for_unknown_signal(signal)

    def wr(self):
        self.memory[self.ar] = self.ac

    def _rase_for_unknown_signal(self, unknown_signal: any) -> None:
        raise RuntimeError(f"Unknown signal: {unknown_signal}")


class Stage(Enum):
    INSTRUCTION_FETCH = 0
    OPERAND_FETCH = 1
    EXECUTE = 2

    def next(self) -> 'Stage':
        return Stage((self.value + 1) % Stage.__len__())


class ControlUnit:
    data_path: DataPath
    start_idx: int
    tick_n: int = 0
    stage: Stage = Stage.INSTRUCTION_FETCH
    decoded_instruction: Optional[Operation] = None

    def __init__(self, data_path: DataPath, start_idx: int):
        self.data_path = data_path
        self.start_idx = start_idx

    def tick(self) -> None:
        if self.stage == Stage.INSTRUCTION_FETCH:
            self.data_path.latch_ar(ArSelSignal.IP)
            self.decoded_instruction = self.data_path.oe()
            self.stage = self.stage.next()
            self.tick_n += 1
        elif self.stage == Stage.OPERAND_FETCH:
            if self.decoded_instruction is None:
                raise RuntimeError("Instruction is not decoded")
            if self.decoded_instruction.arg_type == Operation.ArgType.ADDRESS:
                self.data_path.latch_ar(ArSelSignal.DR)
                self.data_path.latch_dr(DrSelSignal.INSTRUCTION_DECODER)
                self.tick_n += 1
                self.stage = self.stage.next()
            else:
                self.stage = self.stage.next()
                self.tick()
        elif self.stage == Stage.DECODE:




def simulation(program: BinaryProgram, limit: int):


def main(code: str, input: str, output: str) -> None:
    program: BinaryProgram = read_program(code, input)

    simulation(program, limit=1000)


if __name__ == '__main__':
    assert len(sys.argv) == 4, f"Usage: python3 {Path(__file__).name} <code> <input> <output>"
    main(sys.argv[1], sys.argv[2], sys.argv[3])
