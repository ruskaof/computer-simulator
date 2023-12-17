from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from computer_simulator.isa import Arg, ArgType, Opcode, Instruction
from computer_simulator.machine.hardwire import DataPath, ControlUnit, Port

MEMORY_SIZE: int = 2048


@dataclass
class BinaryProgram:
    memory: list[Instruction | int]


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
    memory: list[Instruction | int] = [0 for _ in range(MEMORY_SIZE)]
    for word in json_exe["memory"]:
        arg = None
        if "arg" in word:
            arg = Arg(word["arg"]["value"], ArgType(word["arg"]["type"]))
        comment: str | None = None
        if "comment" in word:
            comment = word["comment"]
        address: int = word["address"]
        memory[address] = Instruction(Opcode(word["opcode"]), arg, comment)
    return BinaryProgram(memory)


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

    result = simulation(program, limit=1_000_000, program_input=program_input)

    print("".join([chr(c) for c in result[0]]))
    print(f"instructions_n: {result[1]} ticks: {result[2]}")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    assert len(sys.argv) >= 3, f"Usage: python3 {Path(__file__).name} <code> <input>"
    main(sys.argv[1], sys.argv[2])
