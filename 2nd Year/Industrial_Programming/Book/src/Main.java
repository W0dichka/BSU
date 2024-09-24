import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;

public class main {
    public static void main(String[] args) throws FileNotFoundException {
        Gson gson = new Gson();

        BufferedReader br = new BufferedReader(new FileReader("input.json"));

        BookShelf A1 = new BookShelf(3);
        Book _json = gson.fromJson(br, Book.class);

        A1.input();
        A1.print();
        System.out.println();

        A1.add(_json);
        A1.print();
        System.out.println();

        A1.Authorsort();
        System.out.println("Список, отсортированный по авторам: ");
        A1.print();

        Book key = new Book("Java", "James Gosling");

        System.out.println("Номер книги 'Java James Gosling': " + A1.binsearchA(key));
        System.out.println();

        A1.Namesort();
        System.out.println("Список, отсортированный по названиям: ");
        A1.print();

        Book key2 = new Book("Python", "Guido van Rossum");
        System.out.println();
        System.out.println("Номер книги 'Python Guido van Rossum': " + A1.binsearchN(key2));
    }
}
