import java.util.Comparator;

public class Book implements Comparable<Book> {
    public String Name;

    public String Author;

    public Book(String name, String author) {
        Name = name;
        Author = author;
    }

    @Override
    public int compareTo(Book book) {
        return Name.compareTo(book.Name);
    }

    public int Authorcompare(Book book) {
        return Author.compareTo(book.Author);
    }

    @Override
    public String toString() {
        return Name + " " + Author;
    }
}