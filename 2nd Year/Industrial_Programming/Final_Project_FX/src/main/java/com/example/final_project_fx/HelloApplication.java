package com.example.final_project_fx;

import javafx.application.Application;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.ChoiceBox;
import javafx.stage.Stage;

import java.io.IOException;

public class HelloApplication extends Application {
    @Override
    public void start(Stage stage) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(HelloApplication.class.getResource("hello-view.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 320, 240);
        stage.setTitle("My Application");
        stage.setScene(scene);
        ObservableList<String> dif_files = FXCollections.observableArrayList( "zip", "xml","txt","json");
        ObservableList<String> dif_actions = FXCollections.observableArrayList( "read", "write","decrypt","encrypt");
        ChoiceBox<String> file_type_choice = new ChoiceBox<String>(dif_files);
        ChoiceBox<String> Action_Choice = new ChoiceBox<String>(dif_actions);
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}