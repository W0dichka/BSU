module com.example.final_project_fx {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires org.kordamp.ikonli.javafx;
    requires java.xml;
    requires com.google.gson;

    opens com.example.final_project_fx to javafx.fxml;
    exports com.example.final_project_fx;
}