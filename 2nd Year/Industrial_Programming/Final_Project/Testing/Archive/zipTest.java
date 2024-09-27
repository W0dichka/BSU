package Archive;

import Factories.ArchiveFactory;
import Factories.FileFactory;
import Files.FileType;
import Files.file;
import org.junit.Assert;
import org.junit.Test;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;
import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class zipTest {
    @Test
    public void Test_zip() throws IOException, ParserConfigurationException, TransformerException, SAXException {
        ArchiveFactory factory = new ArchiveFactory();
        archive zip = factory.getFile(ArchiveType.zip);
        FileFactory factory_text = new FileFactory();
        file txt = factory_text.getFile(FileType.txt);
        String test = "Zip archive";
        new File("test.txt");
        txt.write("test.txt", test);
        zip.write("test.txt");
        new File("result.txt").delete();
        zip.read("test.txt");
        String result = null;
        result = txt.read("test.txt");
        assertEquals(test, result);
    }
}