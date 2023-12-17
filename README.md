# Отчёт по лабораторной работе № 3

Выполнил Русинов Дмитрий Станиславович P33302

Вариант
```lisp | acc | neum | hw | tick | struct | stream | port | pstr | prob1 | 8bit```

Без усложнения

## Язык программирования

Язык lisp-подобный.

Любое выражение в скобках (s-expression) возвращает значение.
Поддерживаются числовые и строковые литералы.

```bnf
program = s_expression

s_expression = "(" atom ")" | expression | "("s_expression")"
   
atomic_symbol = identifier | string_literal | number

expression = defun_expr 
    | if_expr 
    | while_expr 
    | setq_exp
    | print_int_exp
    | user_defined_function_call_exp
    | progn_exp
    
defun_expr = "(" "defun" identifier "(" identifiers ")" s_expression ")"

identifiers = identifier | identifier identifiers

if_expr = "(" "if" s_expression s_expression s_expression ")"

while_expr = "(" "while" s_expression s_expression ")"

setq_exp = "(" "setq" identifier s_expression ")"

print_int_exp = "(" "print_int" s_expression ")"

user_defined_function_call_exp = "(" identifier s_expressions ")"

progn_exp = "(" "progn" s_expressions ")"

s_expressions = s_expression | s_expression s_expressions

identifier = idenitfier_symbol | identifier_symbol identifier

idenitfier_symbol = letter | "_"

string_literal = "\"" *any symbol* "\""
```

## Организация памяти 

После вычисления любого выражения, его результат кладется в аккумулятор.
При вычислении выражения с бинарным оператором, второй операнд вычисляется, кладется на стек,
после чего вычисляется перывый и проводится операция над ними с помощью адресации относительно стека.
