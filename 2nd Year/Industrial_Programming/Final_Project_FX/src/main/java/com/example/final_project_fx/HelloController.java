package com.example.final_project_fx;

import com.example.final_project_fx.Archive.ArchiveType;
import com.example.final_project_fx.Archive.archive;
import com.example.final_project_fx.Encryption.Encryption;
import com.example.final_project_fx.Factories.ArchiveFactory;
import com.example.final_project_fx.Factories.FileFactory;
import com.example.final_project_fx.Files.FileType;
import com.example.final_project_fx.Files.file;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import org.xml.sax.SAXException;

import javax.crypto.BadPaddingException;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;
import java.io.IOException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.spec.ECField;

public class HelloController {

    @FXML
    private Button button;
    @FXML
    private TextField text_field_file_name;
    @FXML
    private TextField text_field;
    @FXML
    private TextField text_input;
    @FXML
    private ChoiceBox Action_Choice;
    @FXML
    private  ChoiceBox file_type_choice;
    @FXML
    protected void click() throws IOException, ParserConfigurationException, SAXException, TransformerException, NoSuchPaddingException, IllegalBlockSizeException, NoSuchAlgorithmException, BadPaddingException, InvalidKeyException {
        String file_type = (String) file_type_choice.getValue();
        String action_choice = (String) Action_Choice.getValue();
        ArchiveFactory A_factory = new ArchiveFactory();
        FileFactory F_factory = new FileFactory();
        Encryption code = new Encryption();
        String file_name = text_field_file_name.getText();
        String text = text_input.getText();
        switch(file_type){
            case "zip":
                archive my_file_zip = A_factory.getFile(ArchiveType.zip);
                switch (action_choice){
                    case "read":
                        my_file_zip.read(file_name);
                        break;
                    case "write":
                        my_file_zip.write(file_name);
                        break;
                    default:
                        throw new IllegalArgumentException("Wrong action");
                }
                break;
            case "txt":
                file my_file_txt = F_factory.getFile(FileType.txt);
                switch (action_choice){
                    case "read":
                        my_file_txt.read(file_name);
                        break;
                    case "write":
                        my_file_txt.write(file_name,text);
                        break;
                    case "decrypt":
                        code.decrypt(file_name,"DOC");
                        break;
                    case "encrypt":
                        code.encrypt(file_name,"DOC");
                        break;
                    default:
                        throw new IllegalArgumentException("Wrong action");
                }
                break;
            case "xml":
                file my_file_xml = F_factory.getFile(FileType.xml);
                switch (action_choice){
                    case "read":
                        my_file_xml.read(file_name);
                        break;
                    case "write":
                        my_file_xml.write(file_name,text);
                        break;
                    default:
                        throw new IllegalArgumentException("Wrong action");
                }
                break;
            case "json":
                file my_file_json = F_factory.getFile(FileType.json);
                switch (action_choice){
                    case "read":
                        my_file_json.read(file_name);
                        break;
                    case "write":
                        my_file_json.write(file_name,text);
                        break;
                    default:
                        throw new IllegalArgumentException("Wrong action");
                }
                break;
            default:
                throw new IllegalArgumentException("Wrong file type");
        }
    }
}