from computer_simulator.translator import ProgramChar, Token, NoValueToken, ValueToken, TranslatorException


def process_whitespace(idx: int, program_chars: list[ProgramChar]) -> int:
    if idx >= len(program_chars):
        return idx
    program_char: ProgramChar = program_chars[idx]
    while idx < len(program_chars) and program_char.char in (" ", "\n", "\t"):
        idx += 1
        if idx < len(program_chars):
            program_char = program_chars[idx]
    return idx


def process_brackets(tokens: list, idx: int, program_chars: list[ProgramChar]) -> int:
    if idx >= len(program_chars):
        return idx
    program_char: ProgramChar = program_chars[idx]
    if program_char.char == "(":
        tokens.append(NoValueToken(NoValueToken.Type.OPEN_BRACKET, program_char))
        idx += 1
    elif program_char.char == ")":
        tokens.append(NoValueToken(NoValueToken.Type.CLOSE_BRACKET, program_char))
        idx += 1
    return idx


def process_binpos(tokens: list, idx: int, program_chars: list[ProgramChar]) -> int:
    if idx >= len(program_chars):
        return idx
    program_char: ProgramChar = program_chars[idx]
    if program_char.char == "+":
        tokens.append(NoValueToken(NoValueToken.Type.BINOP_PLUS, program_char))
        idx += 1
    elif program_char.char == "-":
        tokens.append(NoValueToken(NoValueToken.Type.BINOP_MINUS, program_char))
        idx += 1
    elif program_char.char == "=":
        tokens.append(NoValueToken(NoValueToken.Type.BINOP_EQUAL, program_char))
        idx += 1
    return idx


def process_number_literal(tokens: list, idx: int, program_chars: list[ProgramChar]) -> int:
    if idx >= len(program_chars):
        return idx
    program_char: ProgramChar = program_chars[idx]
    starting_char: ProgramChar = program_char
    if program_char.char.isdigit():
        value = ""
        while idx < len(program_chars) and program_char.char.isdigit():
            value += program_char.char
            idx += 1
            if idx < len(program_chars):
                program_char = program_chars[idx]
        tokens.append(ValueToken(ValueToken.Type.INT, value, starting_char))
    return idx


def process_identifier(tokens: list, idx: int, program_chars: list[ProgramChar]) -> int:
    if idx >= len(program_chars):
        return idx
    program_char: ProgramChar = program_chars[idx]
    starting_char: ProgramChar = program_char
    if program_char.char.isalpha():
        value = ""
        while idx < len(program_chars) and program_char.char.isalpha():
            value += program_char.char
            idx += 1
            if idx < len(program_chars):
                program_char = program_chars[idx]
        tokens.append(ValueToken(ValueToken.Type.IDENTIFIER, value, starting_char))
    return idx


def process_string_literal(tokens: list, idx: int, program_chars: list[ProgramChar]) -> int:
    if idx >= len(program_chars):
        return idx
    program_char: ProgramChar = program_chars[idx]
    starting_char: ProgramChar = program_char
    if program_char.char == '"':
        value = ""
        idx += 1
        if idx < len(program_chars):
            program_char = program_chars[idx]
        while idx < len(program_chars) and program_char.char != '"':
            value += program_char.char
            idx += 1
            if idx < len(program_chars):
                program_char = program_chars[idx]

        if idx >= len(program_chars):
            raise TranslatorException(program_chars[idx - 1], '"')

        tokens.append(ValueToken(ValueToken.Type.STRING, value, starting_char))
        idx += 1
    return idx


def process_booleans(tokens: list, idx: int, program_chars: list[ProgramChar]) -> int:
    if idx >= len(program_chars):
        return idx
    program_char: ProgramChar = program_chars[idx]
    if program_char.char == "T":
        tokens.append(NoValueToken(NoValueToken.Type.TRUE, program_char))
        idx += 1
    elif program_char.char == "F":
        tokens.append(NoValueToken(NoValueToken.Type.FALSE, program_char))
        idx += 1
    return idx


def process_if_statement(tokens: list, idx: int, program_chars: list[ProgramChar]) -> int:
    if idx >= len(program_chars):
        return idx
    program_char: ProgramChar = program_chars[idx]
    next_char = program_chars[idx + 1] if idx + 1 < len(program_chars) else ""
    if program_char.char == "i" and next_char.char == "f":
        tokens.append(NoValueToken(NoValueToken.Type.IF, program_char))
        idx += 2
    return idx


def tokenize(program_chars: list[ProgramChar]) -> list[Token]:
    tokens: list[Token] = []
    idx: int = 0

    while idx < len(program_chars):
        idx = process_whitespace(idx, program_chars)
        idx = process_booleans(tokens, idx, program_chars)
        idx = process_brackets(tokens, idx, program_chars)
        idx = process_binpos(tokens, idx, program_chars)
        idx = process_if_statement(tokens, idx, program_chars)
        idx = process_number_literal(tokens, idx, program_chars)
        idx = process_string_literal(tokens, idx, program_chars)
        idx = process_identifier(tokens, idx, program_chars)

    return tokens
