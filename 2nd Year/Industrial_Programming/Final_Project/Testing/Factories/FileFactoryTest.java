package Factories;

import Files.FileType;
import org.junit.Assert;
import org.junit.Test;

import java.io.File;

import static org.junit.jupiter.api.Assertions.*;

class FileFactoryTest {
    @Test
    public void Test_txt(){
        FileFactory factory = new FileFactory();
        Assert.assertEquals(FileType.txt, factory.getFile(FileType.txt));
    }
    @Test
    public void Test_xml(){
        FileFactory factory = new FileFactory();
        Assert.assertEquals(FileType.xml, factory.getFile(FileType.xml));
    }
    @Test
    public void Test_json(){
        FileFactory factory = new FileFactory();
        Assert.assertEquals(FileType.json, factory.getFile(FileType.json));
    }
}