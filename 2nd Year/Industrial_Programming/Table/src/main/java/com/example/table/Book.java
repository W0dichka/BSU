package com.example.table;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Scanner;

public class Book {
    public String Name;

    public String Author;


    public int Year;

    public Book(String name, String author, int year) {
        Name = name;
        Author = author;
        Year = year;
    }

    public Book(){}
    public int getYear(){
        return Year;
    }
    public void setYear(int year){
        this.Year = year;
    }
    public String getName() {return Name;}
    public String getAuthor()
    {
        return Author;
    }
    public void setName(String name)
    {
        this.Name = name;
    }
    public void setAuthor(String author)
    {
        this.Author = author;
    }
}
