import numpy as np

from pacman_module.game import Agent, Directions

def key(state):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple.
    """

    return (
        state.getPacmanPosition(),
        state.getFood(),
        tuple(state.getCapsules()),
        tuple(state.getGhostStates())
    )

class PacmanAgent(Agent):
    """Empty Pacman agent based on minimax."""

    def __init__(self):
        super().__init__()
        self.explored = dict()

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        # Consider Pacman as MAX player
        return self.minimax(state, 1, 12, -np.inf, +np.inf)[1]

    def minimax(self, state, player: int, depth: int, alpha: float, beta: float):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state:      a game state. See API or class `pacman.GameState`.
            player:     int, 1 means its max's move (pacman), 0 means its min's move (ghost)
            depth:      integer, maximum reachable depth of the search tree
            alpha:      float, the best minimum utility score in the current search tree
            beta:       float, the best maximum utility score in the current search tree

        Returns:
            2-tuple:
                utility score of the current state
                legal move as defined in `game.Directions`
        """

        # Transposition table
        state_key = key(state)
        if state_key in self.explored:
            return self.explored[state_key]

        if state.isWin():
            return 1000 + state.getScore(), Directions.STOP

        if state.isLose():
            return -1000 + state.getScore(), Directions.STOP

        if depth < 0:
            return state.getScore(), Directions.STOP

        # Move initially returned
        move = Directions.STOP

        # Initial best score : -∞ for max player, +∞ for min player
        best_score = -np.inf if player else +np.inf

        for successor, action in state.generatePacmanSuccessors():
            eval = self.minimax(successor, not player, depth - 1, alpha, beta)[0]
            # Max player
            if player:
                # Alpha pruning
                if eval >= beta:
                    return eval, action
                alpha = max(alpha, eval)
                if eval > best_score:
                    best_score = eval
                    move = action

            # Min player
            else:
                # Beta pruning
                if eval <= alpha:
                    return eval, action
                beta = min(beta, eval)

                if eval < best_score:
                    best_score = eval
                    move = action

        # Adding utility score and move of this state to the cache
        self.explored[state_key] = (best_score, move)

        return best_score, move