from enum import Enum, auto


class TokenType(Enum):
    # Specials
    NONE = auto()
    SKIP = auto()

    # Operators
    NOT = auto()
    AND = auto()
    OR = auto()
    ADDITION = auto()
    DIVISION = auto()
    EQUAL = auto()
    GREATER = auto()
    LESS = auto()
    MODULUS = auto()
    MULTIPLICATION = auto()
    NOT_EQUAL = auto()
    SUBTRACTION = auto()

    # Literals
    BOOLEAN = auto()
    RATIONAL = auto()
    INTEGER = auto()
    STRING = auto()

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    SEMICOLON = auto()
    COMMA = auto()

    # Types
    TYPE = auto()

    # Keywords
    VARTYPE = auto()
    CREATE = auto()
    DO = auto()
    ELIF = auto()
    ELSE = auto()
    IF = auto()
    READ = auto()
    SET = auto()
    TO = auto()
    WHILE = auto()
    WRITE = auto()
    END = auto()
    THEN = auto()
    RETURN = auto()
    FUNCTION = auto()

    IDENTIFIER = auto()


STRUCTURE_TOKENS = [
    TokenType.IF,
    TokenType.ELIF,
    TokenType.ELSE,
    TokenType.DO,
    TokenType.WHILE,
    TokenType.FUNCTION
]

OPERATOR_TOKENS = [
    TokenType.NOT,
    TokenType.AND,
    TokenType.OR,
    TokenType.ADDITION,
    TokenType.DIVISION,
    TokenType.EQUAL,
    TokenType.GREATER,
    TokenType.LESS,
    TokenType.MODULUS,
    TokenType.MULTIPLICATION,
    TokenType.NOT_EQUAL,
    TokenType.SUBTRACTION,
]

BOOLEAN_OPERATOR_TOKENS = [
    TokenType.NOT,
    TokenType.AND,
    TokenType.OR,
    TokenType.EQUAL,
    TokenType.GREATER,
    TokenType.LESS,
    TokenType.NOT_EQUAL,
]

NUMERIC_OPERATOR_TOKENS = [
    TokenType.ADDITION,
    TokenType.DIVISION,
    TokenType.MODULUS,
    TokenType.MULTIPLICATION,
    TokenType.SUBTRACTION,
]

LITERAL_TOKENS = [
    TokenType.BOOLEAN,
    TokenType.RATIONAL,
    TokenType.INTEGER,
    TokenType.STRING,
]

INVALID_EXPRESSION_TOKENS = [
    TokenType.TYPE,
    TokenType.VARTYPE,
    TokenType.CREATE,
    TokenType.DO,
    TokenType.ELIF,
    TokenType.ELSE,
    TokenType.IF,
    TokenType.READ,
    TokenType.SET,
    TokenType.TO,
    TokenType.WHILE,
    TokenType.WRITE,
    TokenType.END,
    TokenType.THEN,
    TokenType.RETURN,
    TokenType.FUNCTION
]


class Token:
    def __init__(self, token_type, value, line, column):
        self.token_type = token_type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"T({self.token_type}, {repr(self.value)})"
