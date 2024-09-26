package com.example.final_project_fx.Factories;


import com.example.final_project_fx.Files.*;

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
