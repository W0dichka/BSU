package Files;

import Factories.FileFactory;
import org.junit.Test;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;
import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class jsonTest {
    @Test
    public void Test_json() throws IOException, ParserConfigurationException, TransformerException, SAXException {
        FileFactory factory_text = new FileFactory();
        file json = factory_text.getFile(FileType.json);
        String test = "JSON TEST";
        new File("test.json");
        json.write("test.json", test);
        String result = null;
        result = json.read("test.json");
        assertEquals(test, result);
    }

}