package Files;

import Factories.FileFactory;
import org.junit.Test;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;
import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class xmlTest {
    @Test
    public void Test_xml() throws IOException, ParserConfigurationException, TransformerException, SAXException {
        FileFactory factory_text = new FileFactory();
        file xml = factory_text.getFile(FileType.xml);
        String test = "XML TEST";
        new File("test.xml");
        xml.write("test.xml", test);
        String result = null;
        result = xml.read("test.xml");
        assertEquals(test, result);
    }
}