import java.util.Scanner;

public class test {
public static int somar(int x, int y) {
return x + y;
}
public static int multi(int x, int y) {
return x * y;
}
public static void main(String[] args) {
Scanner scanner = new Scanner(System.in);
int resultado;
int result;
resultado = multi ( 20 , 20 );
result = multi ( 40 , 100 );
System.out.printf("%d - %d", resultado, result);
scanner.close();
}}