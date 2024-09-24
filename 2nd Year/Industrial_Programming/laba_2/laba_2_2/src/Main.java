import java.io.*;
import java.util.Scanner;
import java.util.HashSet;
// 13 Найти максимальное из чисел, встречающихся в заданной матрице более одного раза.

public class Main {
    public static void main(String[] args)  throws Exception
    {
        int sum = 0;
        Scanner in = new Scanner(System.in);
        FileReader gr = new FileReader("input.txt");
        Scanner scan = new Scanner(gr);
        HashSet<Integer> num = new HashSet<Integer>();
        int n = scan.nextInt();
        int max = -2147483648;
        int[][] Matrix = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Matrix[i][j] = scan.nextInt();
                if(num.contains(Matrix[i][j]) == true){
                    if(max<Matrix[i][j]){
                        max = Matrix[i][j];
                    }
                }
                num.add(Matrix[i][j]);
            }
        }
        if(max != -2147483648) {
            System.out.println(max);
        }
        else{
            System.out.println("There are no numbers");
        }
    }
}