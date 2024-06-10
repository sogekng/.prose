from enum import Enum, auto


class VariableType(Enum):
    STRING = auto()
    INTEGER = auto()
    RATIONAL = auto()
    BOOLEAN = auto()


NUMERIC_TYPES = [VariableType.INTEGER, VariableType.RATIONAL]


class Variable:
    def __init__(self, constant: bool, vartype: VariableType, value):
        self.constant = constant
        self.vartype = vartype
        self.value = value


class VariableBank:
    def __init__(self):
        self.scopes: list[dict[str, Variable]] = [{}]

    def start_scope(self) -> None:
        self.scopes.append({})

    def end_scope(self) -> None:
        self.scopes.pop()

    def create(self, name: str, constant: bool, vartype: VariableType, value):
        if name in self.scopes[-1]:
            raise Exception(f"Redeclaration of variable '{name}' in the same scope")

        self.scopes[-1][name] = Variable(constant, vartype, value)

    def exists(self, name: str) -> bool:
        for scope in self.scopes:
            if name in scope:
                return True

        return False

    def is_constant(self, name: str) -> bool:
        return self.get(name).constant

    # NOTE(volatus): "set" conflicts with the built-in function set()
    def redefine(self, name: str, value) -> None:
        variable = self.get(name)

        if variable.constant:
            raise Exception(f"Unable to change value of constant variable '{name}'")

        variable.value = value

    def get(self, name: str) -> Variable:
        if not self.exists(name):
            raise Exception(f"No variable named '{name}' was declared")

        for i in range(len(self.scopes) - 1, -1, -1):
            scope = self.scopes[i]

            if name in scope:
                return scope[name]
