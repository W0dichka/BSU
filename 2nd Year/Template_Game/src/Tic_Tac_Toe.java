public class Tic_Tac_Toe extends Game {
        @Override
        protected String initializeGame(){
                String res = "Tic_Tac_Toe";
                System.out.println("Tic Tac Toe started!");
                System.out.println("Enjoy your game)");
            // tic_tac_toe specific initialization actions
                return res;
        }

        @Override
        protected void playGame() {
            // tic_tac_toe specific play actions
        }

        @Override
        protected void endGame() {
            // tic_tac_toe specific actions to end a game
        }

        @Override
        protected void printWinner() {
            // tic_tac_toe specific actions to print winner
        }

}
