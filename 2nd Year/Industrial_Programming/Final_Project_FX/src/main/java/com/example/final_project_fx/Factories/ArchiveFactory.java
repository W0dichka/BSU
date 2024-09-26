package com.example.final_project_fx.Factories;

import com.example.final_project_fx.Archive.ArchiveType;
import com.example.final_project_fx.Archive.archive;
import com.example.final_project_fx.Archive.zip;

public class ArchiveFactory {

    public archive getFile(ArchiveType type) {
        archive toReturn = null;
        switch (type) {
            case zip:
                toReturn = new zip();
                break;
            default:
                throw new IllegalArgumentException("Wrong archive type:" + type);
        }
        return toReturn;
    }
}
