module com.example.table {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires gson;

    opens com.example.table to javafx.fxml;
    exports com.example.table;
}