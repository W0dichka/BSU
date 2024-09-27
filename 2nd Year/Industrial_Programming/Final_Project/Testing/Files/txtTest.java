package Files;

import Archive.ArchiveType;
import Archive.archive;
import Factories.ArchiveFactory;
import Factories.FileFactory;
import org.junit.Test;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;
import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class txtTest {
    @Test
    public void Test_txt() throws IOException, ParserConfigurationException, TransformerException, SAXException {
        FileFactory factory_text = new FileFactory();
        file txt = factory_text.getFile(FileType.txt);
        String test = "TXT TEST";
        new File("test.txt");
        txt.write("test.txt", test);
        String result = null;
        result = txt.read("test.txt");
        assertEquals(test, result);
    }
}