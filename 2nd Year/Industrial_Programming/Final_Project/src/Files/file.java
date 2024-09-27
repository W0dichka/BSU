package Files;

import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;
import java.io.IOException;

public interface file {
    String read(String file_name) throws IOException, SAXException, ParserConfigurationException;
    void write(String file_name, String text) throws IOException, ParserConfigurationException, SAXException, TransformerException;

}
