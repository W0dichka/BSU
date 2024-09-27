package Archive;


import Files.file;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

public class zip implements archive {
    @Override
    public void read(String file_name) {

        try(ZipInputStream zin = new ZipInputStream(new FileInputStream(file_name)))
        {
            ZipEntry entry;
            String name;
            long size;
            while((entry = zin.getNextEntry())!=null){

                name = entry.getName();
                size = entry.getSize();
                FileOutputStream fout = new FileOutputStream(name);
                for (int c = zin.read(); c != -1; c = zin.read()) {
                    fout.write(c);
                }
                fout.flush();
                zin.closeEntry();
                fout.close();
            }
        }
        catch(Exception ex){

            System.out.println(ex.getMessage());
        }
    }
    @Override
    public void write(String file_name) {
        String zipname = file_name.replace("txt", "zip");

        try(ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(zipname));
            FileInputStream fis = new FileInputStream(file_name)) {

            ZipEntry entry1 = new ZipEntry(file_name);
            zos.putNextEntry(entry1);

            byte[] buffer = new byte[fis.available()];
            fis.read(buffer);

            zos.write(buffer);

            zos.closeEntry();
        }
        catch(Exception e) {
            System.out.println(e.getMessage());
        }
    }
}
