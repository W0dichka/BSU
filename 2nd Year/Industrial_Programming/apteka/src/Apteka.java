import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.stream.Collectors;

public class Apteka {
    public Drug[] shelf;
    public int size;
    public int count;

    public Apteka(int size_apt)
    {
        size = size_apt;
        count = 0;
        shelf = new Drug[size];
    }

    public void input() throws FileNotFoundException{
        Scanner in = new Scanner(new FileReader("input.txt"));
        String name_; String author_; int size_; int date_; int kolvo_; Double price_;
        size_ = in.nextInt();
        size = size_;
        for (int i = 0; i < size; i++) {
            name_ = in.next();
            author_ = in.next();
            kolvo_ = in.nextInt();
            date_ = in.nextInt();
            price_ = in.nextDouble();
            Drug drug = new Drug(name_, author_,price_,kolvo_,date_);
            add(drug);
        }

    }

    public void add(Drug Drug) {
        if (count > size) {
            System.out.println("Shelf is already full.");
            return;
        }
        shelf[count] = Drug;
        count++;
    }

    public Drug pop() {
        Drug temp = shelf[size - 1];
        size--;
        return temp;
    }

    public void Namesort() {
        Arrays.sort(shelf, Drug::compareTo);
    }
    public void Pricesort() {
        Arrays.sort(shelf, Drug::compareTo);
    }

    public void Authorsort() {
        Arrays.parallelSort(shelf, Drug::Authorcompare);
    }

    public int binsearchN(Drug key) {
        return Arrays.binarySearch(shelf, key.Name) + 1;
    }

    public void print() {
        for (int i = 0; i < count; i++)
            System.out.println(shelf[i]);
    }

    public void groupA() throws IOException {
        Map<Object, List<Drug>> group = Arrays.asList(shelf)
                .stream()
                .collect(Collectors.groupingBy(s -> s.Author));
        System.out.println("Books grouped by authors:\n" + group);
    }

    public void filter_(String A) {
        System.out.println("Лекарства, принадлежащие компании " + A + ":");
        List<Drug> filter_ = Arrays.asList(shelf);
        filter_.stream().filter(s -> s.Author.equals(A)).forEach(Drug::print);

    }
}
