package Files;

import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.stream.Collectors;

public class txt implements file{
    @Override
    public String read(String file_name) throws IOException, SAXException, ParserConfigurationException {
        String temp = null;
        FileReader fr = new FileReader(file_name);
        Scanner in = new Scanner(fr);
        while(in.hasNextLine()) {
            temp = temp.concat(in.nextLine());
        }
        in.close();
        return temp;
    }
    @Override
    public void write(String file_name, String text) throws IOException{
        FileWriter fw = new FileWriter(file_name);
        fw.write(text);
        fw.close();
    }
}
