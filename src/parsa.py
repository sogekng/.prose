from dataclasses import dataclass
import sys
import re

from render_method import FunctionBank
from util.token import Token, TokenType, STRUCTURE_TOKENS, LITERAL_TOKENS, INVALID_EXPRESSION_TOKENS, OPERATOR_TOKENS, \
    BOOLEAN_OPERATOR_TOKENS, NUMERIC_OPERATOR_TOKENS
from render import VariableBank, VariableType, Variable, NUMERIC_TYPES

# NOTE(volatus): these exist because Treesitter is stupid
OPEN_BRACKET = "{"
CLOSE_BRACKET = "}"

PRECEDENCE_TABLE = {
    TokenType.OR: 1,
    TokenType.AND: 2,
    TokenType.NOT: 3,

    TokenType.EQUAL: 4,
    TokenType.GREATER: 4,
    TokenType.LESS: 4,
    TokenType.NOT_EQUAL: 4,

    TokenType.ADDITION: 5,
    TokenType.SUBTRACTION: 5,

    TokenType.MULTIPLICATION: 6,
    TokenType.DIVISION: 6,
    TokenType.MODULUS: 6,
}


class Expression:
    def __repr__(self) -> str:
        return f"EXPR"


@dataclass
class ExpressionValue(Expression):
    literal: Token

    def __repr__(self) -> str:
        return f"VALUE({self.literal})"


@dataclass
class ExpressionOperation(Expression):
    operator: Token
    left: ExpressionValue
    right: ExpressionValue

    def __repr__(self) -> str:
        return f"[{self.left}]{self.operator}[{self.right}]"


def match_paren_left_to_right(tokens: list[Token], anchor_index: int) -> int:
    pair_count: int = 0

    for i in range(anchor_index, -1, -1):
        token = tokens[i]

        if token.token_type == TokenType.RPAREN:
            pair_count += 1
        elif token.token_type == TokenType.LPAREN:
            pair_count -= 1

        if pair_count == 0:
            return i

    raise Exception("Unmatched parenthesis in expression")


def parse_expression(tokens: list[Token]) -> Expression:
    if len(tokens) == 0:
        raise Exception("Unexpected empty expression token group")

    if len(tokens) == 1:
        return ExpressionValue(tokens[0])

    if tokens[0].token_type == TokenType.LPAREN and tokens[-1].token_type == TokenType.RPAREN:
        return parse_expression(tokens[1:-1])

    root_index: int = -1
    lowest_precedence: int = sys.maxsize

    i: int = len(tokens) - 1
    while i >= 0:
        token = tokens[i]

        if token.token_type == TokenType.RPAREN:
            i = match_paren_left_to_right(tokens, i)
        elif token.token_type in OPERATOR_TOKENS:
            token_precedence = PRECEDENCE_TABLE[token.token_type]
            if token_precedence < lowest_precedence:
                root_index = i
                lowest_precedence = token_precedence

        i -= 1

    if root_index == -1:
        raise Exception(f"Unexpected extra value '{tokens[1]}' found in expression")

    left_side = tokens[:root_index]

    if len(left_side) == 0:
        raise Exception(f"Missing left side of operator '{tokens[root_index]}'")

    if len(tokens) < root_index + 2:
        raise Exception(f"Missing right side of operator '{tokens[root_index]}'")

    right_side = tokens[root_index + 1:]

    return ExpressionOperation(
        tokens[root_index],
        parse_expression(left_side),
        parse_expression(right_side)
    )


def type_from_literal(literal: Token) -> VariableType:
    if literal.token_type == TokenType.BOOLEAN:
        return VariableType.BOOLEAN
    elif literal.token_type == TokenType.RATIONAL:
        return VariableType.RATIONAL
    elif literal.token_type == TokenType.INTEGER:
        return VariableType.INTEGER
    elif literal.token_type == TokenType.STRING:
        return VariableType.STRING
    else:
        raise Exception(f"Invalid literal type '{literal.token_type}'")


def get_expression_type(expression: Expression, varbank: VariableBank) -> VariableType:
    if isinstance(expression, ExpressionValue):
        if expression.literal.token_type == TokenType.IDENTIFIER:
            variable = varbank.get(expression.literal.value)
            return variable.vartype
        elif expression.literal.token_type in LITERAL_TOKENS:
            return type_from_literal(expression.literal)
        else:
            raise Exception(f"Non-literal and non-identifier toked '{expression.literal}' used as value")
    elif isinstance(expression, ExpressionOperation):
        left_type = get_expression_type(expression.left)
        right_type = get_expression_type(expression.right)

        if expression.operator.token_type in BOOLEAN_OPERATOR_TOKENS:
            if left_type != VariableType.BOOLEAN or right_type != VariableType.BOOLEAN:
                raise Exception(f"Invalid operands for boolean operation")

            return VariableType.BOOLEAN
        elif expression.operator.token_type in NUMERIC_OPERATOR_TOKENS:
            if left_type not in NUMERIC_TYPES or right_type not in NUMERIC_TYPES:
                raise Exception(f"Invalid operands for integer operation")

            if left_type == VariableType.RATIONAL or right_type == VariableType.RATIONAL:
                return VariableType.RATIONAL
            else:
                return VariableType.INTEGER
    else:
        raise Exception(f"Invalid expression type '{expression}'")


def validate_expression(tokens: list, varbank: VariableBank) -> None:
    for token in tokens:
        if token.token_type in INVALID_EXPRESSION_TOKENS:
            raise Exception(f"Invalid token found in expression: '{token}'")

        if token.token_type == TokenType.IDENTIFIER:
            if not varbank.exists(token.value):
                raise Exception(f"No variable named '{token.value}' was declared")

            if varbank.get(token.value).value is None:
                raise Exception(f"Variable declared but unvalued: '{token.value}'")


def validate_expression_method(tokens: list, function_bank: FunctionBank, function_name: str) -> None:
    for token in tokens:
        if token.token_type in INVALID_EXPRESSION_TOKENS:
            raise Exception(f"Invalid token found in expression: '{token}'")

        if token.token_type == TokenType.IDENTIFIER:
            if not function_bank.exists(token.value):
                raise Exception(f"No variable named '{token.value}' was declared")


def lang_type_to_java_type(lang_type: str) -> str:
    if lang_type == VariableType.STRING:
        return "String"
    elif lang_type == VariableType.INTEGER:
        return "int"
    elif lang_type == VariableType.RATIONAL:
        return "float"
    elif lang_type == VariableType.BOOLEAN:
        return "boolean"
    else:
        raise Exception(f"Uknown lang type '{lang_type}'")


def render_expression(tokens: list) -> str:
    return ' '.join([token.value for token in tokens])


def render_parameters(parameters, type_mapping):
    param_tuples = [(parameters[i], parameters[i + 1]) for i in range(0, len(parameters), 2)]
    parameter_string = ", ".join(f"{type_mapping.get(type_, type_)} {name}" for type_, name in param_tuples)

    return parameter_string


def replace_variables(expression: str, values: list) -> str:
    variables = re.findall(r'\b[a-zA-Z]\b', expression)
    concatenated_values = ''.join(values).replace(' ', '')
    value_list = concatenated_values.split(',')

    if len(variables) != len(value_list):
        raise ValueError("O número de variáveis e valores não correspondem.")

    for var, val in zip(variables, value_list):
        expression = re.sub(rf'\b{var}\b', val, expression)

    return expression


@dataclass
class Branch:
    condition_tokens: list
    content_tokens: list

    def __repr__(self) -> str:
        condition = ''
        content = ''

        if self.condition_tokens:
            condition = f"({', '.join([repr(token) for token in self.condition_tokens])})"

        if self.content_tokens:
            content = f"[{', '.join([repr(token) for token in self.content_tokens])}]"

        return f"{condition}=>{content}"


@dataclass
class StructureGroup:
    structure_type: TokenType
    branches: list[Branch]

    def __repr__(self) -> str:
        return f"SG::{self.structure_type}{{{', '.join([repr(branch) for branch in self.branches])}}}"


@dataclass
class StatementGroup:
    tokens: list

    def __repr__(self) -> str:
        return f"({', '.join([repr(token) for token in self.tokens])})"


@dataclass
class Statement:
    tokens: list

    def __repr__(self) -> str:
        return f"<{', '.join([repr(token) for token in self.tokens])}>"

    def validate_syntax(self) -> bool:
        return True

    def render(self, varbank: VariableBank, function_bank: FunctionBank, function_name: str) -> str:
        return ""


class CreateStatement(Statement):
    def __repr__(self) -> str:
        components = []

        if len(self.tokens) >= 1:
            components = self.tokens[1:]

        return f"Create::<{', '.join([repr(token) for token in components])}>"

    def validate_syntax(self) -> bool:
        return (
                len(self.tokens) >= 4
                and self.tokens[0].token_type == TokenType.CREATE
                and self.tokens[1].token_type == TokenType.TYPE
                and self.tokens[2].token_type == TokenType.VARTYPE
                and self.tokens[3].token_type == TokenType.IDENTIFIER
        )

    def render(self, varbank: VariableBank, function_bank: FunctionBank, function_name: str) -> str:
        var_type_str: str = self.tokens[1].value
        is_constant: bool = self.tokens[2].value == 'constant'
        var_name: str = self.tokens[3].value
        var_value = self.tokens[4:] if len(self.tokens) > 4 else None

        var_type: VariableType

        if var_type_str == 'string':
            var_type = VariableType.STRING
        elif var_type_str == 'integer':
            var_type = VariableType.INTEGER
        elif var_type_str == 'rational':
            var_type = VariableType.RATIONAL
        elif var_type_str == 'boolean':
            var_type = VariableType.BOOLEAN
        else:
            raise Exception(f"Unexpected variable type '{var_type_str}'")

        if var_value is not None:
            value_type = get_expression_type(parse_expression(var_value), varbank)

            if value_type != var_type:
                raise Exception(f"Cannot assign value of type {value_type} to identifier of type {var_type}")

        varbank.create(var_name, is_constant, var_type, var_value)

        java_components = []

        if is_constant:
            java_components.append('final')

        java_components.append(lang_type_to_java_type(var_type))
        java_components.append(var_name)

        if var_value is not None:
            java_components.append('=')
            java_components.append(render_expression(var_value))

        return f"{' '.join(java_components)};"


class WriteStatement(Statement):
    def validate_syntax(self) -> bool:
        return (
                len(self.tokens) >= 2
                and self.tokens[0].token_type == TokenType.WRITE
                and (
                        self.tokens[1].token_type == TokenType.STRING
                        or self.tokens[1].token_type == TokenType.IDENTIFIER
                )
        )

    def render(self, varbank: VariableBank, function_bank: FunctionBank, function_name: str) -> str:
        expression_tokens = self.tokens[1:]
        validate_expression(expression_tokens, varbank)
        return f"System.out.printf({', '.join([token.value for token in expression_tokens])});"


class SetStatement(Statement):
    def validate_syntax(self) -> bool:
        return (
                len(self.tokens) >= 4
                and self.tokens[0].token_type == TokenType.SET
                and self.tokens[1].token_type == TokenType.IDENTIFIER
                and self.tokens[2].token_type == TokenType.TO
        )

    def render(self, varbank: VariableBank, function_bank: FunctionBank, function_name: str) -> str:
        expression_tokens = self.tokens[3:]
        var_name = self.tokens[1].value

        if (self.tokens[3].token_type == TokenType.IDENTIFIER
                and self.tokens[4].token_type == TokenType.LPAREN
                and self.tokens[-1].token_type == TokenType.RPAREN):
            validate_expression_method(expression_tokens, function_bank, self.tokens[3].value)

            values = []

            for item in expression_tokens[2:-1]:
                values.append(item.value)

            expression = function_bank.get_function_return_expression(self.tokens[3].value)

            expression = replace_variables(expression, values)

            varbank.redefine(var_name, eval(expression))

            expression = render_expression(expression_tokens)

            return f"{var_name} = {expression};"
        else:
            validate_expression(expression_tokens, varbank)

            varbank.redefine(var_name, expression_tokens)

            return f"{var_name} = {render_expression(expression_tokens)};"


class ReadStatement(Statement):
    def validate_syntax(self) -> bool:
        return (
                len(self.tokens) == 2 and
                self.tokens[0].token_type == TokenType.READ and
                self.tokens[1].token_type == TokenType.IDENTIFIER
        )

    def render(self, varbank: VariableBank, function_bank: FunctionBank, function_name: str) -> str:
        var_name: str = self.tokens[1].value

        variable: Variable = varbank.get(var_name)
        varbank.redefine(var_name, True)

        method: str

        if variable.vartype == VariableType.STRING:
            method = "nextLine"
        elif variable.vartype == VariableType.INTEGER:
            method = "nextInt"
        elif variable.vartype == VariableType.RATIONAL:
            method = "nextFloat"
        elif variable.vartype == VariableType.BOOLEAN:
            method = "nextBoolean"

        return f"{var_name} = scanner.{method}();"


class ReturnStatement(Statement):
    def validate_syntax(self) -> bool:
        return (
                len(self.tokens) >= 2
                and self.tokens[0].token_type == TokenType.RETURN
        )

    def render(self, varbank: VariableBank, function_bank: FunctionBank, function_name: str) -> str:
        expression_tokens = self.tokens[1:]
        # validate_expression_method(expression_tokens, function_bank.get_current_function().variable_bank, function_name)

        expression_rendered = render_expression(expression_tokens)
        function_bank.set_function_return_expression(function_name, expression_rendered)

        return f"return {expression_rendered};"


class CallStatement(Statement):
    def validate_syntax(self) -> bool:
        return (
            len(self.tokens) >= 3
            and self.tokens[0].token_type == TokenType.IDENTIFIER
            and self.tokens[1].token_type == TokenType.LPAREN
            and self.tokens[-1].token_type == TokenType.RPAREN
        )

    def render(self, varbank: VariableBank, function_bank: FunctionBank, function_name: str) -> str:
        function_name_token = self.tokens[0].value
        argument_tokens = self.tokens[2:-1]

        if not function_bank.exists(function_name_token):
            raise Exception(f"Function {function_name_token} is not defined.")

        function = function_bank.get_function(function_name_token)

        arguments = []
        current_arg = []
        for token in argument_tokens:
            if token.token_type == TokenType.COMMA:
                arguments.append(render_expression(current_arg))
                current_arg = []
            else:
                current_arg.append(token)
        if current_arg:
            arguments.append(render_expression(current_arg))

        if len(arguments) != len(function.parameters):
            raise Exception(
                f"Function {function_name_token} expects {len(function.parameters)} arguments, but got {len(arguments)}.")

        rendered_arguments = ", ".join(arguments)
        return f"{function_name_token}({rendered_arguments});"


@dataclass
class Structure:
    branches: list[Branch]

    def __repr__(self) -> str:
        return f"STRUCT::[{', '.join([repr(branch) for branch in self.branches])}]"

    def render(self, varbank: VariableBank, function_bank: FunctionBank, function_name: str) -> list[str]:
        return []

    def render_branch(self, varbank: VariableBank, i: int, function_bank: FunctionBank, function_name: str) -> list[str]:
        lines = []

        varbank.start_scope()

        for item in self.branches[i].content_tokens:

            if isinstance(item, Statement):
                lines.append(item.render(varbank, function_bank, function_name))
            elif isinstance(item, Structure):
                lines.extend(item.render(varbank, function_bank, function_name))
            else:
                raise Exception(f"Unexpected branch item found {item}")

        varbank.end_scope()

        return lines


class IfStructure(Structure):
    def render(self, varbank: VariableBank, function_bank: FunctionBank, function_name: str) -> list[str]:
        lines = []

        lines.append(f"if ({render_expression(self.branches[0].condition_tokens)}) {OPEN_BRACKET}")
        lines.extend(self.render_branch(varbank, 0, function_bank, function_name))

        for i in range(1, len(self.branches)):
            branch = self.branches[i]

            if not branch.condition_tokens:
                lines.append(f"{CLOSE_BRACKET} else {OPEN_BRACKET}")
            else:
                lines.append(f"{CLOSE_BRACKET} else if ({render_expression(branch.condition_tokens)}) {OPEN_BRACKET}")

            lines.extend(self.render_branch(varbank, i, function_bank, function_name))

        lines.append(CLOSE_BRACKET)

        return lines


class WhileStructure(Structure):
    def render(self, varbank: VariableBank, function_bank: FunctionBank, function_name: str) -> list[str]:
        lines = []

        lines.append(f"while ({render_expression(self.branches[0].condition_tokens)}) {OPEN_BRACKET}")
        lines.extend(self.render_branch(varbank, 0, function_bank, function_name))
        lines.append(CLOSE_BRACKET)

        return lines


class DoWhileStructure(Structure):
    def render(self, varbank: VariableBank, function_bank: FunctionBank, function_name: str) -> list[str]:
        lines = []

        lines.append(f"do {OPEN_BRACKET}")
        lines.extend(self.render_branch(varbank, 0, function_bank, function_name))
        lines.append(f"{CLOSE_BRACKET} while ({render_expression(self.branches[0].condition_tokens)});")

        return lines


class FunctionStructure(Structure):
    def render(self, varbank: VariableBank, function_bank: FunctionBank, function_name: str) -> list[str]:
        lines = []

        function_identifier = render_expression(self.branches[0].condition_tokens)
        function_identifier = function_identifier.split()
        function_type = function_identifier[0]
        function_name = function_identifier[1]
        parameters = [item for item in function_identifier[3:-1] if item != ',']

        grouped_parameters = group_parameters(parameters)
        param_definitions = []

        function_bank.create_function(function_name, function_type, grouped_parameters)

        type_mapping = {
            'integer': 'int',
            'string': 'String',
            'boolean': 'bool',
            'rational': 'float'
        }

        for item in grouped_parameters:
            var_type = item[0]
            function_bank.add_variable(function_name, item[1], False, var_type, 0)
            param_definitions.append(f"{type_mapping.get(var_type)} {item[1]}")

        function_type = type_mapping.get(function_type)

        rendered_parameters = ", ".join(param_definitions)

        lines.append(f"public static {function_type} {function_name}({rendered_parameters}) {OPEN_BRACKET}")

        lines.extend(self.render_branch(varbank, 0, function_bank, function_name))

        lines.append(CLOSE_BRACKET)

        return lines


def group_parameters(param_list):
    grouped_params = []

    for i in range(0, len(param_list), 2):
        param_type = param_list[i]
        param_name = param_list[i + 1]
        grouped_params.append((param_type, param_name))
    return grouped_params


def find_next_token(tokens, token_type, offset):
    for i in range(offset, len(tokens)):
        if tokens[i].token_type == token_type:
            return i

    return -1


def group_statement(tokens: list, offset: int) -> StatementGroup:
    statement = []

    statement_end = find_next_token(tokens, TokenType.SEMICOLON, offset)

    if statement_end == -1:
        raise Exception("statement is never finished")

    return StatementGroup(tokens[offset:statement_end])


def group_structures(tokens):
    i = 0

    stack = []
    groups = []

    while i < len(tokens):
        token = tokens[i]

        if token.token_type == TokenType.FUNCTION:
            stack.append(StructureGroup(
                structure_type=token.token_type,
                branches=[Branch(condition_tokens=[], content_tokens=[])]
            ))

            i += 1

            while tokens[i].token_type != TokenType.DO:
                stack[-1].branches[0].condition_tokens.append(tokens[i])
                i += 1

        elif token.token_type == TokenType.DO:
            stack.append(StructureGroup(
                structure_type=token.token_type,
                branches=[Branch(condition_tokens=[], content_tokens=[])]
            ))

            i += 1

            while tokens[i].token_type != TokenType.WHILE:
                stack[-1].branches[0].content_tokens.append(tokens[i])
                i += 1

        elif token.token_type == TokenType.WHILE:
            stack.append(StructureGroup(
                structure_type=token.token_type,
                branches=[Branch(condition_tokens=[], content_tokens=[])]
            ))

            i += 1

            while tokens[i].token_type != TokenType.DO:
                stack[-1].branches[0].condition_tokens.append(tokens[i])
                i += 1

        elif token.token_type == TokenType.IF:
            stack.append(StructureGroup(
                structure_type=token.token_type,
                branches=[Branch(condition_tokens=[], content_tokens=[])]
            ))

            i += 1

            while tokens[i].token_type != TokenType.THEN:
                stack[-1].branches[0].condition_tokens.append(tokens[i])
                i += 1

        elif token.token_type in [TokenType.ELIF, TokenType.ELSE]:
            if not stack or stack[-1].structure_type != TokenType.IF:
                raise Exception(f"{'elif' if token.token_type == TokenType.ELIF else 'else'} used before an if")

            stack[-1].branches.append(Branch(condition_tokens=[], content_tokens=[]))

            if tokens[i].token_type == TokenType.ELIF:
                i += 1

                while tokens[i].token_type != TokenType.THEN:
                    stack[-1].branches[-1].condition_tokens.append(tokens[i])
                    i += 1

        elif token.token_type == TokenType.END:
            if stack[-1].structure_type in STRUCTURE_TOKENS:
                new_group = stack.pop()
                if stack:
                    stack[-1].branches[-1].content_tokens.append(new_group)
                else:
                    groups.append(new_group)

        elif stack and stack[-1].structure_type == TokenType.DO:
            stack[-1].branches[0].condition_tokens.append(token)

        elif stack:
            stack[-1].branches[-1].content_tokens.append(token)

        else:
            groups.append(token)

        i += 1

    return groups


def group_statements(items):
    if not items:
        return items

    groups = []

    i = 0

    while i < len(items):
        item = items[i]

        if isinstance(item, StructureGroup):
            for branch in item.branches:
                branch.content_tokens = group_statements(branch.content_tokens)

            groups.append(item)
            i += 1
        else:
            grouped = group_statement(items, i)
            groups.append(grouped)
            i += len(grouped.tokens) + 1

    return groups


def build_statement(group: StatementGroup) -> Statement:
    statement: Statement

    if group.tokens[0].token_type == TokenType.CREATE:
        statement = CreateStatement(group.tokens)
    elif group.tokens[0].token_type == TokenType.WRITE:
        statement = WriteStatement(group.tokens)
    elif group.tokens[0].token_type == TokenType.READ:
        statement = ReadStatement(group.tokens)
    elif group.tokens[0].token_type == TokenType.SET:
        statement = SetStatement(group.tokens)
    elif group.tokens[0].token_type == TokenType.RETURN:
        statement = ReturnStatement(group.tokens)
    elif group.tokens[0].token_type == TokenType.IDENTIFIER:
        if group.tokens[1].token_type == TokenType.LPAREN and group.tokens[-1].token_type == TokenType.RPAREN:
            statement = CallStatement(group.tokens)
    else:
        raise Exception(f"Unexpected token type '{group.tokens[0].token_type}'")

    if not statement.validate_syntax():
        raise Exception(f"Invalid syntax for token type '{group.tokens[0].token_type}'")

    return statement


def synthesize_statements(items):  # -> list[Structure | Statement]
    elements = []

    i = 0

    while i < len(items):
        item = items[i]

        if isinstance(item, StructureGroup):
            structure: Structure

            if item.structure_type == TokenType.IF:
                structure = IfStructure(branches=[])
            elif item.structure_type == TokenType.DO:
                structure = DoWhileStructure(branches=[])
            elif item.structure_type == TokenType.WHILE:
                structure = WhileStructure(branches=[])
            elif item.structure_type == TokenType.FUNCTION:
                structure = FunctionStructure(branches=[])

            for branch in item.branches:
                structure.branches.append(Branch(
                    condition_tokens=branch.condition_tokens,
                    content_tokens=synthesize_statements(branch.content_tokens)
                ))

            elements.append(structure)
        elif isinstance(item, StatementGroup):
            if len(item.tokens) == 0:
                raise Exception("Unexpected empty statement group")

            elements.append(build_statement(item))
        else:
            raise Exception(f"Unexpected ungrouped element {item}")

        i += 1

    return elements
