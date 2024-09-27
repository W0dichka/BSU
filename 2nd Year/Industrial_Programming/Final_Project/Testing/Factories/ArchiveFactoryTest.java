package Factories;

import Archive.ArchiveType;
import Archive.archive;
import Archive.zip;
import org.junit.Assert;
import org.junit.Test;

import static org.junit.jupiter.api.Assertions.*;

class ArchiveFactoryTest {
    @Test
    public void Test_zip(){
        ArchiveFactory factory = new ArchiveFactory();
        Assert.assertEquals(ArchiveType.zip, factory.getFile(ArchiveType.zip));
    }
    @Test(expected = IllegalArgumentException.class)
    public void Test_error(){
        ArchiveFactory factory = new ArchiveFactory();
        Assert.assertEquals(ArchiveType.zip, factory.getFile(ArchiveType.rar));
    }
}