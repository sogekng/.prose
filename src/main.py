from lexer import Lexer
from parsa import *
from render import VariableBank
import pprint
import sys
from os import path
import subprocess

EXTENSION = "prose"
VERSION = "1.0.3"
USAGE = "Usage: lang <input_file>"


def main():
    print(f"Lang v{VERSION}")

    if len(sys.argv) < 2:
        print(f"Missing argument <file.{EXTENSION}>")
        print(USAGE)
        return

    file_path = sys.argv[1]

    if not file_path.endswith(f".{EXTENSION}"):
        print(f"File has the wrong extension; should end with '.{EXTENSION}'")
        return

    path_dirname = path.dirname(file_path)
    path_basename = path.basename(file_path)

    if len(path_basename) < len(EXTENSION) + 2:
        print("Invalid file name provided")
        return

    program_name = path_basename[:-len(EXTENSION) - 1]
    output_path = path.join(path_dirname, f"{program_name}.java")

    try:
        code: str

        try:
            with open(file_path, 'r') as file:
                code = file.read()
        except FileNotFoundError:
            print(f"O arquivo {file_path} não foi encontrado.")
            return

        lexer = Lexer()
        tokens = lexer.tokenize(code)

        # print("TOKENS::::")
        # pprint.pp(tokens)
        grouped_tokens = group_statements(group_structures(tokens))
        # print("GROUPED TOKENS::::")
        # pprint.pp(grouped_tokens)
        syntax_tree = synthesize_statements(grouped_tokens)
        # print("SYNTAX TREE::::")
        # pprint.pp(syntax_tree)

        print("Output:", output_path)

        varbank = VariableBank()
        lines = []

        for syntax_item in syntax_tree:
            if isinstance(syntax_item, Statement):
                lines.append(syntax_item.render(varbank))
            elif isinstance(syntax_item, Structure):
                lines.extend(syntax_item.render(varbank))
            else:
                raise Exception(f"Unexpected syntax item '{syntax_item}'")

        java_code = "\n".join(lines)

        # print("Variaveis:\n")
        # pprint.pp(varbank.scopes)
        # print()

        with open(output_path, 'w') as output_file:
            # Preamble
            output_file.write(f"import java.util.Scanner;\n\n")
            output_file.write(f"public class {program_name} {{\n")
            output_file.write("public static void main(String[] args) {\n")
            output_file.write("Scanner scanner = new Scanner(System.in);\n")

            # Código gerado dinamicamente
            output_file.write(java_code)

            # Epilogue
            output_file.write("\nscanner.close();\n")
            output_file.write("}}\n")

        compile_command = ["javac", output_path]
        subprocess.run(compile_command, check=True)
        print(f"Arquivo {output_path} compilado com sucesso.")

        # Executar o arquivo Java compilado
        execute_command = ["java", "-cp", path_dirname, program_name]
        result = subprocess.run(execute_command, check=True, capture_output=True, text=True)

        print("Saída da execução do programa Java:\n")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao compilar ou executar o arquivo Java: {e}")
        print(f"Saída do erro: {e.output}")
    except Exception as e:
        print(f"Ocorreu um erro ao processar o arquivo: {e}")


if __name__ == "__main__":
    main()
