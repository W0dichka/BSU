package com.example.final_project_fx.Encryption;

import com.example.final_project_fx.Factories.FileFactory;
import com.example.final_project_fx.Files.FileType;
import com.example.final_project_fx.Files.file;
import org.xml.sax.SAXException;

import javax.crypto.*;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;
import java.io.File;
import java.io.IOException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;

public class Encryption {
    public void encrypt(String file_name,String algo) throws NoSuchAlgorithmException, NoSuchPaddingException, IOException, ParserConfigurationException, SAXException, IllegalBlockSizeException, BadPaddingException, TransformerException {
        KeyGenerator keygenerator = KeyGenerator.getInstance(algo);
        SecretKey myDesKey = keygenerator.generateKey();
        // Creating object of Cipher
        Cipher desCipher;
        desCipher = Cipher.getInstance("DES");
        FileFactory factory_text = new FileFactory();
        file txt = factory_text.getFile(FileType.txt);
        String temp = txt.read(file_name);
        // Creating byte array to store string
        byte[] text = temp.getBytes("UTF8");
        byte[] textEncrypted = desCipher.doFinal(text);
        String s = new String(textEncrypted);
        new File("encrypted.txt");
        txt.write("encrypted.txt",s);
    }
    public void decrypt(String file_name, String algo) throws NoSuchAlgorithmException, NoSuchAlgorithmException, NoSuchPaddingException, IOException, InvalidKeyException, ParserConfigurationException, SAXException, IllegalBlockSizeException, BadPaddingException, TransformerException {
        KeyGenerator keygenerator = KeyGenerator.getInstance(algo);
        SecretKey myDesKey = keygenerator.generateKey();
        Cipher desCipher = null;
        FileFactory factory_text = new FileFactory();
        file txt = factory_text.getFile(FileType.txt);
        // Decrypting text
        desCipher.init(Cipher.ENCRYPT_MODE, myDesKey);
        String text = txt.read(file_name);
        byte[] textEncrypted = desCipher.doFinal(text.getBytes());
        String s = new String(textEncrypted);
        new File("decrypted.txt");
        txt.write("decrypted.txt",s);
    }
}
