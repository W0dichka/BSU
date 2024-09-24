import java.util.Arrays;

public class Book {

    public Kod[] list;

    public int size;

    public int count;

    public Book(int size_)
    {
        size = size_;
        count = 0;
        list = new Kod[size];
    }

    public void add(Kod Kod) {
        list[count] = Kod;
        count++;
    }

    public Kod pop() {
        Kod temp = list[size - 1];
        size--;
        return temp;
    }

    public void Namesort() {
        Arrays.sort(list, Kod::compareTo);
    }
    public int binsearchN(Kod key) {
        return Arrays.binarySearch(list, key.number) + 1;
    }

    public void print() {
        for (int i = 0; i < count; i++)
            System.out.println(list[i]);
    }
}
