import java.io.*;
import java.util.Scanner;
//3.  Выведите номера столбцов, все элементы, которых четны.
// Для каждого столбца с  отрицательным  элементом  на главной диагонали
// вывести его номер и сумму всех элементов этого столбца.
public class Main {
    public static void main(String[] args)  throws Exception
    {
        int sum = 0;
        Scanner in = new Scanner(System.in);
        FileReader gr = new FileReader("input.txt");
        Scanner scan = new Scanner(gr);
        int n = scan.nextInt();
        int[][] Matrix = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                Matrix[i][j] = scan.nextInt();
            }
        int k=0;
        int flag = 0;
        System.out.print("Column with even numbers: " );
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Matrix[j][i] % 2 != 0) {
                    flag = 1;
                    j = n;
                }
            }
            if (flag != 1){
                System.out.print(i+1);
                System.out.print(" ");
                k++;
            }
            flag = 0;
        }
        if(k==0){
            System.out.print(" There is no column with even numbers");
        }
        k=0;
        System.out.println("");
        System.out.println("Column with negative numbers :" );
        for(int i = 0; i < n; i++){
            if(Matrix[i][i] < 0)
            {
                System.out.print("Number = " );
                System.out.print(i+1);
                System.out.print(", ");
                System.out.print("Sum = " );
                for(int j = 0; j < n; j++){
                    sum += Matrix[j][i];
                }
                k++;
                System.out.println(sum);
                sum =0;
            }
        }
        if(k==0){
            System.out.println("There is no column with negative numbers");
        }
    }
}