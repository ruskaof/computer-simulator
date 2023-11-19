import pytest

from computer_simulator.translator import ProgramChar, NoValueToken, ValueToken, TranslatorException
from computer_simulator.translator.tokenizer import tokenize


def test_tokenize_whitespace_only():
    program_chars = [ProgramChar(" ", 1, 1), ProgramChar("\n", 2, 1), ProgramChar("\t", 3, 1)]
    assert tokenize(program_chars) == []


def test_tokenize_brackets():
    program_chars = [ProgramChar("(", 1, 1), ProgramChar(")", 1, 2)]
    assert tokenize(program_chars) == [NoValueToken(NoValueToken.Type.OPEN_BRACKET, ProgramChar("(", 1, 1)),
                                       NoValueToken(NoValueToken.Type.CLOSE_BRACKET, ProgramChar(")", 1, 2))]


def test_tokenize_operators():
    program_chars = [ProgramChar("+", 1, 1), ProgramChar("-", 1, 2)]
    assert tokenize(program_chars) == [NoValueToken(NoValueToken.Type.BINOP_PLUS, ProgramChar("+", 1, 1)),
                                       NoValueToken(NoValueToken.Type.BINOP_MINUS, ProgramChar("-", 1, 2))]


def test_tokenize_digits():
    program_chars = [ProgramChar("1", 1, 1), ProgramChar("2", 1, 2), ProgramChar("3", 1, 3)]
    assert tokenize(program_chars) == [ValueToken(ValueToken.Type.INT, "123", ProgramChar("1", 1, 1))]


def test_tokenize_alpha():
    program_chars = [ProgramChar("a", 1, 1), ProgramChar("b", 1, 2), ProgramChar("c", 1, 3)]
    assert tokenize(program_chars) == [ValueToken(ValueToken.Type.IDENTIFIER, "abc", ProgramChar("a", 1, 1))]


def test_tokenize_quotes():
    program_chars = [ProgramChar('"', 1, 1), ProgramChar("a", 1, 2), ProgramChar('"', 1, 3)]
    assert tokenize(program_chars) == [ValueToken(ValueToken.Type.STRING, "a", ProgramChar('"', 1, 1))]


def test_tokenize_unterminated_string():
    program_chars = [ProgramChar('"', 1, 1), ProgramChar("a", 1, 2)]
    with pytest.raises(TranslatorException) as e:
        tokenize(program_chars)
    assert str(e.value) == 'Expected " at line 1, column 2. Got a'


def test_tokenize_lisp_program():
    program_chars = [
        ProgramChar('(', 1, 1), ProgramChar('s', 1, 2), ProgramChar('e', 1, 3), ProgramChar('t', 1, 4),
        ProgramChar('q', 1, 5),
        ProgramChar(' ', 1, 6), ProgramChar('a', 1, 7), ProgramChar(' ', 1, 8), ProgramChar('2', 1, 9),
        ProgramChar(')', 1, 10),
        ProgramChar(' ', 1, 11),
        ProgramChar('(', 1, 12), ProgramChar('p', 1, 13), ProgramChar('r', 1, 14), ProgramChar('i', 1, 15),
        ProgramChar('n', 1, 16), ProgramChar('t', 1, 17),
        ProgramChar(' ', 1, 18),
        ProgramChar('(', 1, 19), ProgramChar('i', 1, 20), ProgramChar('f', 1, 21), ProgramChar(' ', 1, 22),
        ProgramChar('(', 1, 23), ProgramChar('=', 1, 24), ProgramChar(' ', 1, 25), ProgramChar('a', 1, 26),
        ProgramChar(' ', 1, 27), ProgramChar('2', 1, 28), ProgramChar(')', 1, 29),
        ProgramChar(' ', 1, 30), ProgramChar('1', 1, 31), ProgramChar('0', 1, 32), ProgramChar('0', 1, 33),
        ProgramChar(')', 1, 34), ProgramChar(')', 1, 35)
    ]
    expected_tokens = [
        NoValueToken(NoValueToken.Type.OPEN_BRACKET, ProgramChar('(', 1, 1)),
        ValueToken(ValueToken.Type.IDENTIFIER, "setq", ProgramChar('s', 1, 2)),
        ValueToken(ValueToken.Type.IDENTIFIER, "a", ProgramChar('a', 1, 7)),
        ValueToken(ValueToken.Type.INT, "2", ProgramChar('2', 1, 9)),
        NoValueToken(NoValueToken.Type.CLOSE_BRACKET, ProgramChar(')', 1, 10)),
        NoValueToken(NoValueToken.Type.OPEN_BRACKET, ProgramChar('(', 1, 12)),
        ValueToken(ValueToken.Type.IDENTIFIER, "print", ProgramChar('p', 1, 13)),
        NoValueToken(NoValueToken.Type.OPEN_BRACKET, ProgramChar('(', 1, 19)),
        NoValueToken(NoValueToken.Type.IF, ProgramChar('i', 1, 20)),
        NoValueToken(NoValueToken.Type.OPEN_BRACKET, ProgramChar('(', 1, 23)),
        NoValueToken(NoValueToken.Type.BINOP_EQUAL, ProgramChar('=', 1, 24)),
        ValueToken(ValueToken.Type.IDENTIFIER, "a", ProgramChar('a', 1, 26)),
        ValueToken(ValueToken.Type.INT, "2", ProgramChar('2', 1, 28)),
        NoValueToken(NoValueToken.Type.CLOSE_BRACKET, ProgramChar(')', 1, 29)),
        ValueToken(ValueToken.Type.INT, "100", ProgramChar('1', 1, 31)),
        NoValueToken(NoValueToken.Type.CLOSE_BRACKET, ProgramChar(')', 1, 34)),
        NoValueToken(NoValueToken.Type.CLOSE_BRACKET, ProgramChar(')', 1, 35))
    ]
    assert tokenize(program_chars) == expected_tokens
