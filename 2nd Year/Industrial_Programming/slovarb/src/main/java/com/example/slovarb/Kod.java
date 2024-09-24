package com.example.slovarb;
import java.util.Comparator;

public class Kod{

    int number;

    String word;

    public Kod(int number_, String word_)
    {
        number = number_;
        word = word_;
    }

    public int Numbercompare(Kod kod) {
        if(number > kod.number)
            return 1;
        else if(number < kod.number)
            return -1;
        else
            return 0;
    }
}
