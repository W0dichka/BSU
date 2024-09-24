import java.io.*;
import java.util.Scanner;
import java.util.HashSet;
// 23 Найти максимум среди всех локальных минимумов заданной матрицы.
// Элемент матрицы называется локальным максимумом, если он строго больше всех имеющихся у него соседей.
// Соседями элемента ajj в матрице назовем элементы aki  с i-1ki+1, j-1lj+1,(k,l)(i,j).

public class Main {
    public static void main(String[] args)  throws Exception
    {
        int sum = 0;
        Scanner in = new Scanner(System.in);
        FileReader gr = new FileReader("input.txt");
        Scanner scan = new Scanner(gr);
        int n = scan.nextInt();
        int[] max = new int[n*n];
        int[][] Matrix = new int[n+2][n+2];
        for (int i = 0; i < n+2; i++) {
            for (int j = 0; j < n+2; j++) {
                Matrix[i][j] = 2147483647;
            }
        }
        int p = 0;
        int flag = 0;
        for (int i = 1; i < n+1; i++) {
            for (int j = 1; j < n + 1; j++) {
                Matrix[i][j] = scan.nextInt();
            }
        }
        for (int i = 1; i < n+1; i++) {
            for (int j = 1; j < n+1; j++) {
                for(int k = i - 1; k <= i + 1; k++) {
                    for (int l = j - 1; l <= j + 1; l++) {
                        if ((k != i) && (l != j) && (Matrix[i][j] >= Matrix[k][l])) {
                            flag++;
                        }
                    }
                }
                        if (flag == 0) {
                            max[p] = Matrix[i][j];
                            p++;
                        }
                        flag = 0;
            }
        }
        int maximum = max[0];
        for(int i = 1; i < p; i++){
            if(maximum < max[i]){
                maximum = max[i];
            }
        }
        System.out.println(Integer.toString(maximum));
    }
}