from computer_simulator.translator import Token

IDENTIFIER_NON_ALPHA_CHARS = {"_"}


def process_whitespace(idx: int, chars: str) -> int:
    if idx >= len(chars):
        return idx
    while idx < len(chars) and chars[idx] in (" ", "\n", "\t"):
        idx += 1
    return idx


def process_brackets(tokens: list, idx: int, chars: str) -> int:
    if idx >= len(chars):
        return idx
    if chars[idx] == "(":
        tokens.append(Token(Token.Type.OPEN_BRACKET, chars[idx]))
        idx += 1
    elif chars[idx] == ")":
        tokens.append(Token(Token.Type.CLOSE_BRACKET, chars[idx]))
        idx += 1
    return idx


def process_binpos(tokens: list, idx: int, chars: str) -> int:
    if idx >= len(chars):
        return idx
    if chars[idx] in ("+", "-", "*", "/", "=", "<", ">"):
        tokens.append(Token(Token.Type.BINOP, chars[idx]))
        idx += 1
    return idx


def process_number_literal(tokens: list, idx: int, chars: str) -> int:
    if idx >= len(chars):
        return idx
    start_idx = idx
    while idx < len(chars) and chars[idx].isdigit():
        idx += 1
    if idx > start_idx:
        tokens.append(Token(Token.Type.INT, chars[start_idx:idx]))
    return idx


def process_identifier(tokens: list, idx: int, chars: str) -> int:
    if idx >= len(chars):
        return idx
    start_idx = idx
    while idx < len(chars) and chars[idx].isalpha():
        idx += 1
    if idx > start_idx:
        tokens.append(Token(Token.Type.IDENTIFIER, chars[start_idx:idx]))
    return idx


def process_string_literal(tokens: list, idx: int, chars: str) -> int:
    if idx >= len(chars):
        return idx
    if chars[idx] == '"':
        value = ""
        idx += 1
        while idx < len(chars) and chars[idx] != '"':
            value += chars[idx]
            idx += 1

        if idx >= len(chars):
            raise RuntimeError("Expected closing quote")

        tokens.append(Token(Token.Type.STRING, value))
        idx += 1
    return idx


def process_booleans(tokens: list, idx: int, chars: str) -> int:
    if idx >= len(chars):
        return idx
    if chars[idx] in ("T", "F"):
        tokens.append(Token(Token.Type.BOOLEAN, chars[idx]))
        idx += 1
    return idx


def process_identifier_like_statement(tokens: list, idx: int, chars: str, token_type: Token.Type, keyword: str) -> int:
    if idx >= len(chars):
        return idx
    value = ""
    end_idx = idx
    while end_idx < len(chars) and (chars[end_idx].isalpha() or chars[end_idx] in IDENTIFIER_NON_ALPHA_CHARS):
        value += chars[end_idx]
        end_idx += 1
    if value == keyword:
        tokens.append(Token(token_type, keyword))
        return end_idx

    return idx


def tokenize(program_chars: str) -> list[Token]:
    tokens: list[Token] = []
    idx: int = 0

    while idx < len(program_chars):
        prev_idx = idx

        prev_idx = process_whitespace(prev_idx, program_chars)
        prev_idx = process_booleans(tokens, prev_idx, program_chars)
        prev_idx = process_brackets(tokens, prev_idx, program_chars)
        prev_idx = process_binpos(tokens, prev_idx, program_chars)
        prev_idx = process_identifier_like_statement(tokens, prev_idx, program_chars, Token.Type.IF, "if")
        prev_idx = process_identifier_like_statement(tokens, prev_idx, program_chars, Token.Type.SETQ, "setq")
        prev_idx = process_identifier_like_statement(tokens, prev_idx, program_chars, Token.Type.DEFUN, "defun")
        prev_idx = process_identifier_like_statement(tokens, prev_idx, program_chars, Token.Type.PRINT_CHAR, "print_char")
        prev_idx = process_identifier_like_statement(tokens, prev_idx, program_chars, Token.Type.PRINT_STRING, "print_string")
        prev_idx = process_identifier_like_statement(tokens, prev_idx, program_chars, Token.Type.PROGN, "progn")
        prev_idx = process_identifier_like_statement(tokens, prev_idx, program_chars, Token.Type.READ_STRING, "read_string")
        prev_idx = process_identifier_like_statement(tokens, prev_idx, program_chars, Token.Type.WHILE, "while")
        prev_idx = process_number_literal(tokens, prev_idx, program_chars)
        prev_idx = process_string_literal(tokens, prev_idx, program_chars)
        prev_idx = process_identifier(tokens, prev_idx, program_chars)

        if prev_idx == idx:
            raise RuntimeError(f"Unknown token: {program_chars[prev_idx]}")
        idx = prev_idx

    return tokens
