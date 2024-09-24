package com.example.slovarb;

import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;

public class HelloController {

    @FXML
    private Button button;

    @FXML
    private TextField textfield_1;

    @FXML
    private TextField textfield_2;

    @FXML
    protected void click() throws Exception {
        int x;
        String final_str = "";
        String null_str = "-";
        Book slovar = new Book(10);
        slovar.input();
        slovar.Numbersort();
        String str = textfield_1.getText();
        if(str == ""){
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Warning");
            alert.setHeaderText(null);
            alert.setContentText("Input numbers and try again!");
            alert.showAndWait();
        }
        else{
            String[] subStr;
            String delimeter = " ";
            subStr = str.split(delimeter);
            for (int i = 0; i < subStr.length; i++) {
                x = Integer.parseInt(subStr[i]);
                int j = slovar.binsearchN(x);
                if (j == -1) {
                    final_str += null_str;
                } else {
                    final_str += slovar.list[j].word;
                }
            }
            textfield_2.setText(final_str);
        }
    }
}