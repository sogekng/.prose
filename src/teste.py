import re


def find_identifiers(expression):
    # Encontrar todos os identificadores
    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)
    return identifiers


# Testando a função com seu código
code = """
create string variable leia;
function string ler(string ops, string opa) do
    return ops + opa;
end
set leia to ler("Ola mundo", "again");
write "%s" leia;
"""

print(find_identifiers(code))
