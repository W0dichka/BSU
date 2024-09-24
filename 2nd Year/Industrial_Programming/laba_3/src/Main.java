import java.io.*;
import java.util.Scanner;

//3.  В тексте после буквы Р, если она не последняя в слове,
// ошибочно напечатана буква А вместо О. Внести исправления в текст.
public class Main {
    public static void main(String[] args) throws Exception
    {
        FileWriter fr = new FileWriter("output.txt");
        FileReader gr = new FileReader("input.txt");
        Scanner scan = new Scanner(gr);
        String text[];
        text = new String[100];
        int i = 0;
        //reading
        while(scan.hasNextLine())
        {
            text[i] = scan.nextLine();
            i++;
        }
        int j = 0;
        int x=0;
        for(int k = 0; k < i; k++)
        {
            text[k] =  text[k].replace("pa", "po");
            text[k] =  text[k].replace("Pa", "Po");
            text[k] =  text[k].replace("PA", "PO");
            text[k] =  text[k].replace("pA", "pO");
        }
        //writing
        while(j < i)
        {
            fr.write(text[j] + "\n");
            fr.write(1);
            j++;
        }
        gr.close();
        fr.close();
        System.out.println("Succede, check output.txt!");
    }
}