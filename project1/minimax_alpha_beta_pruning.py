import numpy as np
import psutil
import os

from pacman_module.game import Agent, Directions
from functools import lru_cache

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
            # Cache info
        cache_info = self.minimax.cache_info()

        # Memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        print("Cache Stats:")
        print(f"Hits: {cache_info.hits}")
        print(f"Misses: {cache_info.misses}")
        print(f"Current Cache Size: {cache_info.currsize}")

        print("\nMemory Usage:")
        print(f"RSS (Resident Set Size): {memory_info.rss / 1024**2:.2f} MB")
        print(f"VMS (Virtual Memory Size): {memory_info.vms / 1024**2:.2f} MB")

        return self.minimax(state, 1, 12, -np.inf, +np.inf)[1]

    @lru_cache(maxsize=4096)
    def minimax(self, state, maxPlayer:bool, maxDepth:int, alpha:float, beta:float):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state:      a game state. See API or class `pacman.GameState`.
            maxPlayer:  boolean, 1 means its max's move, 0 means its min's move
            maxDepth:   integer, maximum reachable depth of the search tree
            alpha:      float, best minimum utility score in the current search tree
            beta:       float, best maximum utility score in the current search tree

        Returns:
            tuple of 4 elements
                - utility score of the current state
                - legal move as defined in `game.Directions`
        """

        # Testing if the game is won or finished
        if state.isWin():
            return 1000 + state.getScore(), Directions.STOP

        if state.isLose():
            return -1000 + state.getScore(), Directions.STOP

        if maxDepth < 0:
            return state.getScore(), Directions.STOP

        # Move initially returned
        move = Directions.STOP

        # Case of the MAX player
        if maxPlayer:
            # Max score
            rv_max = - np.inf
            # Getting the legal actions for the MAX player (Pacman)
            for successor, action in state.generatePacmanSuccessors():
                eval = self.minimax(successor, 0, maxDepth-1, alpha, beta)[0]

                # Alpha pruning
                if eval >= beta:
                    return eval, action
                alpha = max(alpha, eval)

                if eval > rv_max:
                    rv_max = eval
                    move = action

            return rv_max, move

        else:
            # Min score
            rv_min = + np.inf
            # Getting the legal actions for the MIN player (Ghost)
            for successor, action in state.generateGhostSuccessors(1):
                eval = self.minimax(successor, 1, maxDepth-1, alpha, beta)[0]

                # Beta pruning
                if eval <= alpha:
                    return eval, action
                beta = min(beta, eval)

                if eval < rv_min:
                    rv_min = eval
                    move = action

            return rv_min, move
