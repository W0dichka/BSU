import java.util.Comparator;
public class Drug implements Comparable<Drug> {
    public String Name;
    public String Author;
    public Double Price;

    public int Date;

    public int Kolvo;

    public Drug(String name, String author, Double price_, int kolvo_, int date_ ) {
        Name = name;
        Author = author;
        Price = price_;
        Kolvo = kolvo_;
        Date = date_;
    }
    @Override
    public int compareTo(Drug drug) {
        return Name.compareTo(drug.Name);
    }

    public int Authorcompare(Drug drug) {
        return Author.compareTo(drug.Author);
    }

    public int Namecompare(Drug drug) {
        return Name.compareTo(drug.Name);
    }

    public int Pricecompare(Drug drug) {
        if(Price > drug.Price)
            return -1;
        else if(Price < drug.Price)
            return 1;
        else
            return 0;
    }
    public void print() {
        System.out.print("Название лекарства : " + Name + " Срок годности : " + Date);
        System.out.print(" Название фирмы производителя : " + Author);
        System.out.print(" Количество упаковок " + Kolvo + "Ценв упаковки" + Price);
    }
}
