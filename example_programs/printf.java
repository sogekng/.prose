import java.util.Scanner;

public class printf
{
    public static void main(String[] args)
    {
        Scanner scanner = new Scanner(System.in);

        final String str_world = "World";
        System.out.printf("Hello %s!", str_world);
        scanner.close();
    }
}
