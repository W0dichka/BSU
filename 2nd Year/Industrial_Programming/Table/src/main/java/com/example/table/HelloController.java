package com.example.table;
import javafx.collections.ObservableList;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.cell.PropertyValueFactory;
import java.io.*;
import java.util.Scanner;


public class HelloController {
    ObservableList<Book> books = FXCollections.observableArrayList();
    @FXML
    private  Button button_1;
    @FXML
    private TableView<Book> Table_1;
    @FXML
    private TableColumn<Book, String> imya;
    @FXML
    private TableColumn<Book, Integer> god;
    @FXML
    private TableColumn<Book, String> avtor;
    @FXML
    protected void click() throws Exception{
        Scanner in = new Scanner(new FileReader("input.txt"));
        int size =in.nextInt();
        for (int i = 0; i < size; i++) {
            String name_ = in.next();
            String avtor_ = in.next();
            int year = in.nextInt();
            books.add(new Book(name_,avtor_,year));
        }
        avtor.setCellValueFactory(new PropertyValueFactory<Book, String>("Name"));
        god.setCellValueFactory(new PropertyValueFactory<Book, Integer>("Year"));
        imya.setCellValueFactory(new PropertyValueFactory<Book, String>("Author"));
        Table_1.setItems(books);
    }
}c