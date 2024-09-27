package Files;

import com.google.gson.Gson;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;

public class json implements file{
    @Override
    public String read(String file_name) throws IOException, SAXException, ParserConfigurationException {
        BufferedReader br = new BufferedReader(new FileReader(file_name));
        String result = new Gson().fromJson(br, String.class);
        br.close();
        return result;
    }

    @Override
    public void write(String file_name, String text) throws IOException {
        FileWriter fw = new FileWriter(file_name);
        fw.write(new Gson().toJson(text));
        fw.close();
    }
}
