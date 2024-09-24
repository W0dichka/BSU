import java.io.*;
import java.text.*;
import java.util.Scanner;
//[-1,1)
public class Main {

    public static void main(String[] args) {
        System.out.println(new String("Input x and k"));
        Scanner in = new Scanner(System.in);
        double x = in.nextDouble();
        int k = in.nextInt();
        in.close();
        double result = 0;
        double result2 = 0;
        double e = Math.pow(10,-2*k);
        double temp  = -x;
        int i = 1;
        while(Math.abs(temp)>e) {
            result += temp;
            temp = temp * x * i / (i+1);
            i++;
        }
        NumberFormat formatter = NumberFormat.getNumberInstance();
        formatter.setMaximumFractionDigits(k);
        System.out.println("My Math");
        System.out.println(formatter.format(result));
        System.out.println("Java Math");
        result2 = Math.log(1-x);
        System.out.println(formatter.format(result2));
    }
}