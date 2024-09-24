public class Monopoly extends Game {

    @Override
    protected String initializeGame() {
        String res = "Monopoly";
        System.out.println("Monopoly started!");
        System.out.println("Enjoy your game)");
        // monopoly specific initialization actions
        return res;
    }

    @Override
    protected void playGame() {
        // monopoly specific play actions
    }

    @Override
    protected void endGame() {
        // monopoly specific actions to end a game
    }

    @Override
    protected void printWinner() {
        // monopoly specific actions to print winner
    }

}