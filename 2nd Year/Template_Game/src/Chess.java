public class Chess extends Game {

    @Override
    protected String initializeGame() {
        String res = "Chess";
        System.out.println("Chess started!");
        System.out.println("Enjoy your game)");
        // chess specific initialization actions
        return res;
    }

    @Override
    protected void playGame() {
        // chess specific play actions
    }

    @Override
    protected void endGame() {
        // chess specific actions to end a game
    }

    @Override
    protected void printWinner() {
        // chess specific actions to print winner
    }

}
