import pytest
from computer_simulator.translator import Token
from computer_simulator.translator.tokenizer import tokenize


def test_whitespace_processing_handles_empty_input():
    result = tokenize("")
    assert result == []


def test_whitespace_processing_handles_only_whitespace():
    result = tokenize("   \n\t  ")
    assert result == []

def test_number_literal_processing_handles_non_numeric_input():
    result = tokenize("abc")
    assert result == [Token(Token.Type.IDENTIFIER, "abc")]


def test_identifier_processing_handles_non_alpha_input():
    result = tokenize("123")
    assert result == [Token(Token.Type.INT, "123")]


def test_string_literal_processing_handles_unterminated_string():
    with pytest.raises(RuntimeError):
        tokenize('"abc')


def test_booleans_processing_handles_non_boolean_input():
    result = tokenize("abc")
    assert result == [Token(Token.Type.IDENTIFIER, "abc")]


def test_if_statement_processing_handles_non_if_input():
    result = tokenize("abc")
    assert result == [Token(Token.Type.IDENTIFIER, "abc")]


def test_tokenize_handles_complex_input():
    result = tokenize('(= if ("abc") aaa eee)')
    assert result == [
        Token(Token.Type.OPEN_BRACKET, "("),
        Token(Token.Type.BINOP, "="),
        Token(Token.Type.IF, "if"),
        Token(Token.Type.OPEN_BRACKET, "("),
        Token(Token.Type.STRING, "abc"),
        Token(Token.Type.CLOSE_BRACKET, ")"),
        Token(Token.Type.IDENTIFIER, "aaa"),
        Token(Token.Type.IDENTIFIER, "eee"),
        Token(Token.Type.CLOSE_BRACKET, ")"),
    ]

def test_tokenize_handles_complex_input2():
    result = tokenize("(+ (1) (2))")
    assert result == [
        Token(Token.Type.OPEN_BRACKET, "("),
        Token(Token.Type.BINOP, "+"),
        Token(Token.Type.OPEN_BRACKET, "("),
        Token(Token.Type.INT, "1"),
        Token(Token.Type.CLOSE_BRACKET, ")"),
        Token(Token.Type.OPEN_BRACKET, "("),
        Token(Token.Type.INT, "2"),
        Token(Token.Type.CLOSE_BRACKET, ")"),
        Token(Token.Type.CLOSE_BRACKET, ")"),
    ]
