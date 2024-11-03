import numpy as np
from pacman_module.game import Agent, Directions


def key(state):
    return (
        state.getPacmanPosition(),
        tuple(state.getGhostStates()),
        state.getFood()
    )


class PacmanAgent(Agent):
    """Empty Pacman agent based on minimax."""

    def __init__(self):
        super().__init__()
        self.cache = dict()
        self.max_depth = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        # Automatic max depth estimation, using the game size
        if self.max_depth is None:
            self.max_depth = np.ceil(
                np.log(state.getWalls().width * state.getWalls().height)
                / np.log(4)
            )

        return self.minimax(state, 1, 4, -np.inf, +np.inf, set())[1]

    def minimax(self, state, player: int, depth: int, alpha: float,
                beta: float, _explored: set):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state:      a game state. See API or class
                        `pacman.GameState`.
            player:     int, 1 means its max's move (pacman),
                        0 means its min's move (ghost)
            depth:      integer, maximum reachable depth of the search tree
            alpha:      float, the best minimum utility score in the current
                        search tree
            beta:       float, the best maximum utility score in the current
                        search tree
            _explored:  set of already explored states in the current node,
                        prevents cycles

        Returns:
            2-tuple:
                utility score of the current state
                legal move as defined in `game.Directions`
        """
        # Check game end
        if state.isWin() or state.isLose():
            return state.getScore(), None

        # Check transposition table
        state_key = key(state)
        if (state_key, state.getScore()) in self.cache:
            return self.cache[(state_key, state.getScore())]

        # Move initially returned
        move = Directions.STOP

        if depth < 0:
            return state.getScore(), move

        # Check cut-off condition
        if depth < 0:
            return state.getScore(), move

        # Initial best score : -∞ for pacman, +∞ for ghost
        best_score = -np.inf if player else +np.inf

        # Update explored set (copy is necessary: python sucks)
        explored = _explored.copy()
        explored.add(state_key)

        # Determine successors based on player
        successors = (
            state.generatePacmanSuccessors() if player
            else state.generateGhostSuccessors(1)
        )

        # Explore successor nodes
        for successor, action in successors:
            # Check if successor is already explored
            if key(successor) in _explored:
                continue

            eval = self.minimax(successor, not player, depth - 1,
                                alpha, beta, explored)[0]

            # Max player (pacman)
            if player:
                # Best score / best move update
                if eval > best_score:
                    best_score = eval
                    move = action

                # Alpha pruning
                if best_score >= beta:
                    break
                alpha = max(alpha, best_score)

            # Min player (ghost)
            else:
                # Best score / best move update
                if eval < best_score:
                    best_score = eval
                    move = action

                # Beta pruning
                if best_score <= alpha:
                    break
                beta = min(beta, best_score)

        # Adding entry in transposition table
        self.cache[(state_key, state.getScore())] = (best_score, move)

        return best_score, move