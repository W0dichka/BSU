package Factories;


import Archive.ArchiveType;
import Archive.archive;
import Archive.zip;

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
