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

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        # Consider Pacman as MAX player
        return self.minimax(state, True, 12, -np.inf, +np.inf, set())[1]

    def minimax(self, state, player: bool, depth: int, alpha: float,
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

        # Move initially returned
        move = Directions.STOP

        # Check game end
        if state.isWin() or state.isLose():
            return state.getScore(), move

        # Check transposition table
        state_key = key(state)
        if (state_key, state.getScore()) in self.cache:
            return self.cache[(state_key, state.getScore())]

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
            if key(successor) in explored:
                continue

            eval = self.minimax(successor, not player, depth - 1,
                                alpha, beta, explored)[0]

            # Max player (pacman)
            if player:
                alpha_pruning = eval >= beta
                better_score = eval > best_score

                # Best score / best move update
                if alpha_pruning or better_score:
                    best_score = eval
                    move = action

                # Alpha pruning
                if alpha_pruning:
                    break

                alpha = max(alpha, eval)

            # Min player (ghost)
            else:
                beta_pruning = eval <= alpha
                better_score = eval < best_score

                # Best score / best move update
                if beta_pruning or better_score:
                    best_score = eval
                    move = action

                # Beta pruning
                if beta_pruning:
                    break

                beta = min(beta, eval)

        # Adding entry in transposition table
        self.cache[(state_key, state.getScore())] = (best_score, move)

        return best_score, move
