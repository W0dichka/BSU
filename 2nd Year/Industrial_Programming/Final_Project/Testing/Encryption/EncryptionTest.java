package Encryption;

import Factories.FileFactory;
import Files.FileType;
import Files.file;
import org.junit.Test;
import org.xml.sax.SAXException;

import javax.crypto.BadPaddingException;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;
import java.io.File;
import java.io.IOException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;

import static org.junit.jupiter.api.Assertions.*;

class EncryptionTest {
    @Test
    public void Test_encrypt() throws IOException, ParserConfigurationException, TransformerException, SAXException, NoSuchPaddingException, IllegalBlockSizeException, NoSuchAlgorithmException, BadPaddingException, InvalidKeyException {
        FileFactory factory_text = new FileFactory();
        file txt = factory_text.getFile(FileType.txt);
        String test = "TXT TEST";
        new File("test.txt");
        txt.write("test.txt", test);
        String result = null;
        Encryption code = new Encryption();
        code.encrypt("test.txt","dec");
        code.decrypt("test.txt","dec");
        result = txt.read("decrypted.txt");
        assertEquals(test, result);
    }
}