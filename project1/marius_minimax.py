import numpy as np
from pacman_module.game import Agent, Directions


class PacmanAgent(Agent):
    """Empty Pacman agent based on minimax."""

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        # Consider Pacman as MAX player
        return self.minimax(state, 1, 20)[1]

    def minimax(self, state, maxplayer, maxDepth):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.
            MAXplayer: boolean value that indicates if the player is the MAX player or not

        Returns:
            A tuple of 2 elements:
                - The score of the minimax
                - A legal move as defined in `game.Directions`.
        """

        # Testing if the game is won or finished
        if state.isWin():
            return 1000 + state.getScore(), Directions.STOP

        elif state.isLose():
            return -1000 + state.getScore(), Directions.STOP

        elif maxDepth < 0:
            return state.getScore(), Directions.STOP

        # Move initially returned
        move = Directions.STOP

        # Case of the MAX player
        if maxplayer:
            # Max score
            rv_max = - np.inf
            # Getting the legal actions for the MAX player (Pacman)
            for successor, action in state.generatePacmanSuccessors():
                eval = self.minimax(successor, 0, maxDepth-1)[0]

                if eval > rv_max:
                    rv_max = eval
                    move = action

            return rv_max, move

        else:
            # Min score
            rv_min = + np.inf
            # Getting the legal actions for the MIN player (Ghost)
            for successor, action in state.generateGhostSuccessors(1):
                eval = self.minimax(successor, 1, maxDepth-1)[0]

                if eval < rv_min:
                    rv_min = eval
                    move = action

            return rv_min, move
