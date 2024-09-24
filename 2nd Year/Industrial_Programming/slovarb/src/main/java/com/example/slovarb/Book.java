package com.example.slovarb;
import java.util.Arrays;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.stream.Collectors;

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
    public int binsearchN(int key) {
        int firstIndex = 0;
        int lastIndex = list.length - 1;

        // условие прекращения (элемент не представлен)
        while(firstIndex <= lastIndex) {
            int middleIndex = (firstIndex + lastIndex) / 2;
            // если средний элемент - целевой элемент, вернуть его индекс
            if (list[middleIndex].number == key) {
                return middleIndex;
            }

            // если средний элемент меньше
            // направляем наш индекс в middle+1, убирая первую часть из рассмотрения
            else if (list[middleIndex].number < key)
                firstIndex = middleIndex + 1;

                // если средний элемент больше
                // направляем наш индекс в middle-1, убирая вторую часть из рассмотрения
            else if (list[middleIndex].number > key)
                lastIndex = middleIndex - 1;

        }
        return -1;
    }


    public void print() {
        for (int i = 0; i < count; i++)
            System.out.println(list[i].number);
    }
    public void input() throws FileNotFoundException {
        Scanner in = new Scanner(new FileReader("input.txt"));
        String word_;
        int number_;
        int size_;
        int x;
        size_ = in.nextInt();
        size = size_;
        for (int i = 0; i < size; i++) {
            number_ = in.nextInt();
            word_ = in.next();
            Kod kod = new Kod(number_, word_);
            add(kod);
        }
    }
    public void Numbersort() {
        Arrays.sort(list, Kod::Numbercompare);
    }
}
