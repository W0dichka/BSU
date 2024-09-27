import Archive.ArchiveType;
import Archive.archive;
import Factories.ArchiveFactory;
import Factories.FileFactory;
import Files.*;

import java.io.File;
import java.util.Scanner;

public class Main
{
    private static final String txt_to_zip = "test_write.txt";
    private static final String zip_to_txt = "test_read.zip";
    public static void main(String[] args) throws Exception {
        ArchiveFactory factory = new ArchiveFactory();
        archive zip = factory.getFile(ArchiveType.zip);
        FileFactory factory_text = new FileFactory();
        file txt = factory_text.getFile(FileType.txt);
        zip.read(zip_to_txt);
        zip.write(txt_to_zip);
    }
}