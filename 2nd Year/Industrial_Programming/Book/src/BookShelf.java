import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Scanner;

public class BookShelf {
    public Book[] shelf;
    public int size;
    public int count;

    public BookShelf() {
        size = 0;
        count = 0;
        shelf = new Book[size];
    }

    public BookShelf(int size_) {
        size = size_;
        count = 0;
        shelf = new Book[size];
    }

    public void input() throws FileNotFoundException {
        Scanner in = new Scanner(new FileReader("input.txt"));

        String Name; String Author; int size_;

        size_ = in.nextInt();

        size = size_;

        for (int i = 0; i < size; i++) {
            Name = in.next();
            Author = in.nextLine();

            Book book = new Book(Name, Author);

            add(book);
        }
    }

    public void add(Book book) {
        if (count > size) {
            System.out.println("Shelf is already full.");
            return;
        }
        shelf[count] = book;
        count++;
    }

    public Book pop() {
        Book temp = shelf[size - 1];
        size--;
        return temp;
    }

    public void Namesort() {
        Arrays.sort(shelf, Book::compareTo);
    }

    public void Authorsort() {
        Arrays.parallelSort(shelf, Book::Authorcompare);
    }

    public int binsearchA(Book key) {
        return Arrays.binarySearch(shelf, key.Author) + 1;
    }

    public int binsearchN(Book key) {
        return Arrays.binarySearch(shelf, key.Name) + 1;
    }

    public void print() {
        for (int i = 0; i < count; i++)
            System.out.println(shelf[i]);
    }
}

