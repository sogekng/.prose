import re
from util.token import Token, TokenType

token_regex = {
    # Specials
    TokenType.SKIP: re.compile(r'([\s\n]+)|#.*'),

    # Operators
    TokenType.NOT: re.compile(r'!'),
    TokenType.AND: re.compile(r'&&'),
    TokenType.OR: re.compile(r'\|\|'),
    TokenType.ADDITION: re.compile(r'\+'),
    TokenType.DIVISION: re.compile(r'/'),
    TokenType.EQUAL: re.compile(r'=='),
    TokenType.GREATER: re.compile(r'>'),
    TokenType.LESS: re.compile(r'<'),
    TokenType.MODULUS: re.compile(r'%'),
    TokenType.MULTIPLICATION: re.compile(r'\*'),
    TokenType.NOT_EQUAL: re.compile(r'!='),
    TokenType.SUBTRACTION: re.compile(r'-'),

    # Literals
    TokenType.BOOLEAN: re.compile(r'\btrue|false\b'),
    TokenType.RATIONAL: re.compile(r'\d+\.\d+|\.\d+'),
    TokenType.INTEGER: re.compile(r'\d+'),
    TokenType.STRING: re.compile(r'"([^"\\]|\\.)*"'),

    # Delimiters
    TokenType.LPAREN: re.compile(r'\('),
    TokenType.RPAREN: re.compile(r'\)'),
    TokenType.SEMICOLON: re.compile(r';'),
    TokenType.COMMA: re.compile(r','),

    # Types
    TokenType.TYPE: re.compile(r'\b(string)|(integer)|(rational)|(boolean)\b'),

    # Keywords
    TokenType.VARTYPE: re.compile(r'\b(constant)|(variable)\b'),
    TokenType.CREATE: re.compile(r'\bcreate\b'),
    TokenType.DO: re.compile(r'\bdo\b'),
    TokenType.ELSE: re.compile(r'\belse\b'),
    TokenType.ELIF: re.compile(r'\belif\b'),
    TokenType.IF: re.compile(r'\bif\b'),
    TokenType.READ: re.compile(r'\bread\b'),
    TokenType.SET: re.compile(r'\bset\b'),
    TokenType.TO: re.compile(r'\bto\b'),
    TokenType.WHILE: re.compile(r'\bwhile\b'),
    TokenType.WRITE: re.compile(r'\bwrite\b'),
    TokenType.END: re.compile(r'\bend\b'),
    TokenType.THEN: re.compile(r'\bthen\b'),
    TokenType.RETURN: re.compile(r'\breturn\b'),
    TokenType.FUNCTION: re.compile(r'\bfunction\b'),

    TokenType.IDENTIFIER: re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)'),
}

class Lexer:
    """
    Classe Lexer para realizar a análise léxica (tokenização).

    Atributos:
        tokens (list): Lista de tokens identificados.
        current_char (str): Caractere atual sendo analisado.
        current_position (int): Posição atual do caractere no texto.
        current_line (int): Linha atual no texto.
        current_column (int): Coluna atual no texto.
        text (str): Texto completo que está sendo analisado.
        token_specs (list): Especificações de tokens com nomes e expressões regulares.
        token_regex (list): Lista de tuplas (tipo de token, expressão regular compilada).
    """
    def __init__(self):
        """
        Inicializa o analisador léxico com especificações de tokens e prepara o ambiente de análise.
        """
        self.reset()


    def reset(self, new_text=""):
        self.current_position = 0
        self.current_line = 1
        self.current_column = 0
        self.tokens = []
        self.current_char = None
        self.text = new_text
        self.current_char = self.text[0] if self.text else None

    
    def tokenize(self, text):
        """
        Processa o texto para extrair tokens conforme as especificações.

        Args:
            text (str): Texto a ser tokenizado.

        Return:
            list: Lista de objetos Token gerados a partir do texto.
        """
        self.reset(text)

        token = None
        while (token := self.next_token()) is not None:
            if token.token_type != TokenType.SKIP:
                self.tokens.append(token)

        return self.tokens


    def next_token(self):
        """
        Identifica o próximo token no texto baseado nas especificações regulares.

        Return:
            Token: O próximo token identificado ou None se um token deve ser ignorado (como espaços).
        """
        if self.current_char is None:
            # return Token('EOF', '', self.current_line, self.current_column)
            return None

        if self.current_char == '\n':
            token = Token(TokenType.SKIP, self.current_char, self.current_line, self.current_column)
            self.advance()
            return token

        for token_type, regex in token_regex.items():
            match = regex.match(self.text, self.current_position)

            if not match:
                continue

            lexeme = match.group(0)
            token = Token(token_type, lexeme, self.current_line, self.current_column)
            self.advance(len(lexeme))
            return token

        self.advance()
        return None


    def advance(self, steps=1):
        """
        Avança o caractere atual no texto, atualizando posição, linha e coluna.

        Args:
            steps (int): Número de passos para avançar no texto.

        Return:
            Token: O próximo token identificado ou None se um token deve ser ignorado (como espaços).
        """
        for _ in range(steps):
            if self.current_position + 1 >= len(self.text):
                self.current_char = None
            else:
                self.current_position += 1
                self.current_char = self.text[self.current_position]

                if self.current_char == '\n':
                    self.current_line += 1
                    self.current_column = 0
                else:
                    self.current_column += 1
