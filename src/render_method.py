from render import *


class Function:
    def __init__(self, name, return_type, parameters):
        self.name = name
        self.return_type = return_type
        self.parameters = parameters
        self.variable_bank = VariableBank()
        self.return_expression = None

    def set_return_expression(self, expression):
        self.return_expression = expression

    def get_return_expression(self):
        return self.return_expression


class FunctionBank:
    def __init__(self):
        self.functions = {}
        self.current_function_stack = []

    def create_function(self, name, return_type, parameters):
        if self.exists(name):
            raise Exception(f"Function {name} is already defined.")
        new_function = Function(name, return_type, parameters)
        self.functions[name] = new_function

    def get_function(self, name):
        if not self.exists(name):
            raise Exception(f"Function {name} does not exist.")
        return self.functions[name]

    def exists(self, name):
        return name in self.functions

    def start_scope(self, function_name):
        if not self.exists(function_name):
            raise Exception(f"Function {function_name} is not defined.")
        self.current_function_stack.append(self.functions[function_name])

    def end_scope(self):
        if not self.current_function_stack:
            raise Exception("No function scope to end.")
        self.current_function_stack.pop()

    def get_current_function(self):
        if not self.current_function_stack:
            raise Exception("No function scope is currently active.")
        return self.current_function_stack[-1]

    def add_variable(self, function_name, name, constant, vartype, value=None):
        if not self.exists(function_name):
            raise Exception(f"Function {function_name} does not exist.")
        self.functions[function_name].variable_bank.create(name, constant, vartype, value)

    def get_variable(self, function_name, name):
        if not self.exists(function_name):
            raise Exception(f"Function {function_name} does not exist.")
        return self.functions[function_name].variable_bank.get(name)

    def set_variable(self, function_name, name, value):
        if not self.exists(function_name):
            raise Exception(f"Function {function_name} does not exist.")
        self.functions[function_name].variable_bank.redefine(name, value)

    def set_function_return_expression(self, function_name, expression):
        if not self.exists(function_name):
            raise Exception(f"Function {function_name} does not exist.")
        self.functions[function_name].set_return_expression(expression)

    def get_function_return_expression(self, function_name):
        if not self.exists(function_name):
            raise Exception(f"Function {function_name} does not exist.")
        return self.functions[function_name].get_return_expression()
