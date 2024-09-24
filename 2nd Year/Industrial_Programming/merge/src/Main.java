import java.util.ArrayList;

public class Main {


    void merge(ArrayList<Integer> a, ArrayList<Integer> b) {
        int i = 0;
        int j = 0;
        ArrayList c = new ArrayList(b.size() + a.size());
        while(i < a.size() && j < b.size()){
            if (a.get(i) <= b.get(j)) {
                c.add(a.get(i)); i++;
            }
            else {
                c.add(b.get(j)); j++;
            }
        }
        while(i < a.size()) {
            c.add(a.get(i));
            i++;
        }
        while(j < b.size()){
            c.add(b.get(j));
            j++;
        }
        a = c;
    }






    public static void main(String[] args) {
        System.out.println("Hello world!");

    }
}