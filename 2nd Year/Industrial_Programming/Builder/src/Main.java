public class Main {
    public static void main(String[] args) {
        Person  myPerson = new Person.Builder()
                .withName("Kirill")
                .withSurname("Kanashevich")
                .withAge(19)
                .withHeight(184)
                .withWeight(73)
                .build();
    }
}