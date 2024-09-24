module com.example.sum {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;

    opens com.example.sum to javafx.fxml;
    exports com.example.sum;
}