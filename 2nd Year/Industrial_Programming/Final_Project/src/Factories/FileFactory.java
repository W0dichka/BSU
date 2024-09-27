package Factories;

import Archive.zip;
import Files.*;

public class FileFactory {

    public file getFile(FileType type) {
        file toReturn = null;
        switch (type) {
            case txt:
                toReturn = new txt();
                break;
            case json:
                toReturn = new json();
                break;
            case xml:
                toReturn = new xml();
                break;
            default:
                throw new IllegalArgumentException("Wrong file type:" + type);
        }
        return toReturn;
    }
}
