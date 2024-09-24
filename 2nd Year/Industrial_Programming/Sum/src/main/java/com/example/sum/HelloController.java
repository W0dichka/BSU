package com.example.sum;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;

public class HelloController {
    @FXML
    private Button button;
    @FXML
    private TextField textfield_1;
    @FXML
    private TextField textfield_2;
    @FXML
    private TextField textfield_3;
    @FXML
    protected void add() {
        String s1;
        String s2;
        s1 = textfield_1.getText();
        s2 = textfield_2.getText();
        double chislo_1 = Double.parseDouble(s1);
        double chislo_2 = Double.parseDouble(s2);
        double sum = chislo_1 + chislo_2;
        String result = Double.toString(sum);
        textfield_3.setText(result);
    }
}