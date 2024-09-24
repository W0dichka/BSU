module com.example.slovarb {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;

    opens com.example.slovarb to javafx.fxml;
    exports com.example.slovarb;
}