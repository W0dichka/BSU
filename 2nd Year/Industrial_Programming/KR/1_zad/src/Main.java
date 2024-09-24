import java.io.*;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws Exception {
        FileWriter fr = new FileWriter("output.txt");
        FileReader gr = new FileReader("input.txt");
        Scanner scan = new Scanner(gr);
        int x1,x2;
        int k=1;
        x1 = scan.nextInt();
        if(x1 == 0)
        {
            fr.write("0");
            gr.close();
            fr.close();
            return;
        }
        x2 = scan.nextInt();
        if(x2>x1)
        {
            while(x2 != 0) {
                if(x2>x1){
                    k++;
                    x1 = x2;
                    x2 = scan.nextInt();
                }
                else {
                    fr.write(Integer.toString(k));
                    gr.close();
                    fr.close();
                    return;
                }
            }
        }
        else if(x2<x1) {
            while(x2 != 0) {
                if (x1 < x2) {
                    k++;
                }
                x2 = x1;
                x1 = scan.nextInt();
            }
        }
        fr.write(Integer.toString(k));
        gr.close();
        fr.close();
    }
}