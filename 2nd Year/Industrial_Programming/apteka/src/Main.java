import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws Exception {
        FileWriter fr = new FileWriter("output.txt");
        FileReader gr = new FileReader("input.txt");
        Apteka shop = new Apteka(3);
        shop.input();
        shop.print();
        System.out.println();

        shop.Namesort();
        System.out.println("Список, отсортированный по названияю лекарств: ");
        shop.print();

        Drug key = new Drug("Лиофилизат","Кедрион", 22.0, 200, 2022);
        System.out.println("Номер лекарства : " + shop.binsearchN(key));
        System.out.println();

        shop.Pricesort();
        System.out.println("Список, отсортированный по убыванию цены лекарств: ");
        shop.print();
        System.out.println();

        shop.groupA();
        shop.filter_("Кедрион");

    }
}