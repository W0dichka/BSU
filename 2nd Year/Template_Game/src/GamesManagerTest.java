import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GamesManagerTest {
    @Test
    public void TestMonopoly(){
        Game game;
        game = new Monopoly();
        assertEquals("Monopoly", game.initializeGame());
    }
    @Test
    public void TestChess() {
        Game game;
        game = new Chess();
        assertEquals("Chess", game.initializeGame());
    }
    @Test
    public void TestTic() {
        Game game;
        game = new Tic_Tac_Toe();
        assertEquals("Tic_Tac_Toe", game.initializeGame());
    }
}