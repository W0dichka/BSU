import java.io.*;
import java.util.Scanner;
import java.util.StringTokenizer;
import java.util.regex.*;

public class Main {
    public static void main(String[] args) throws Exception {
        Scanner in = new Scanner(System.in);
        String str = in.next();
        double final_string = 0;
        StringTokenizer st = new StringTokenizer(str, "\n ?!,");
        while (st.hasMoreTokens()) {
            String tmp = st.nextToken();
            if (tmp.matches("[-]?[0-9]+[.][0-9]+[fd]?")) {
                final_string += Double.parseDouble(tmp);
                System.out.println(tmp);

            } else if (tmp.matches("[-]?[0-9][.][0-9]+e[+-][0-9]+")) {
                final_string += Double.parseDouble(tmp);
                System.out.println(tmp);

            } else if(tmp.matches("([-]?[0-9]+[.]?[0-9]*)")) {
                final_string += Double.parseDouble(tmp);
                System.out.println(tmp);
            }
        }
        System.out.println(final_string);
    }
}