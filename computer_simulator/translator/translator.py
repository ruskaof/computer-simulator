from __future__ import annotations

import sys
from pathlib import Path

from computer_simulator.translator import Token
from computer_simulator.translator.expression_executor import translate_expression, Program, translate_program
from computer_simulator.translator.tokenizer import tokenize


def run_translator(tokens: list[Token]) -> Program:
    program: Program = Program()
    translate_program(tokens, program)
    return program


def read_file(filename: str) -> str:
    return Path(filename).read_text(encoding="utf-8")


def main(source: str, target: str) -> None:
    with open(target, "w", encoding="utf-8") as f:
        source_code: str = read_file(source)
        tokenized_code: list[Token] = tokenize(source_code)
        program: Program = run_translator(tokenized_code)
        print(f"source LoC: {len(source_code.split("\n"))} code instr: {len(program.operations)}")
        f.write(program.to_machine_code())


if __name__ == "__main__":
    assert len(sys.argv) == 3, f"Usage: python3 {Path(__file__).name} <source> <target>"
    main(sys.argv[1], sys.argv[2])
