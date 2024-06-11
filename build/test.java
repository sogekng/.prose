import java.util.Scanner;

public class test {
public static int somar(int x, int y) {
return x + y;
}
public static String ler(String leia) {
return leia;
}
public static void main(String[] args) {
Scanner scanner = new Scanner(System.in);
String leia;
int resultado;
leia = ler ( "Ola mundo" );
resultado = somar ( 10 , 20 );
System.out.printf("%s, o resultado e: %d", leia, resultado);
scanner.close();
}}