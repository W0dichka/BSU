import java.io.*;
import java.text.*;
import java.util.Scanner;

public class GamesManager {

    public static void main (String [] args) throws  Exception{
        Scanner in = new Scanner(System.in);
        System.out.println("Enter name of Game");
        System.out.println("Chess = 1");
        System.out.println("Monopoly = 2");
        System.out.println("Tic Tac Toe = 3");
        String Game_Mode = in.next();

        Game game;

        switch (Game_Mode){
            case "1" :
                game = new Chess();
                break;
            case "2" :
                game = new Monopoly();
                break;
            case "3":
                game = new Tic_Tac_Toe();
                break;
            default :
                throw new IllegalStateException();
        }

        game.playOneGame(2);
    }

}
