from pacman_module.game import Agent, Directions
from pacman_module.util import Stack, PriorityQueue

import numpy as np


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
    )


class PacmanAgent(Agent):
    """Pacman agent based on astar search."""

    def __init__(self):
        super().__init__()

        self.moves = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        if self.moves is None:
            self.moves = self.astar(state)

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def astar(self, state):

        path = []
        fringe = PriorityQueue()

        first_score = 0 + state.getNumFood()

        fringe.push((state, path, first_score), first_score)
        closed = set()

        # cost = length(path)
        # heuristic = state.getNumFood()

        while True:
            if fringe.isEmpty():
                print("Fringe empty", path)
                return path

            priority, (current, path, score) = fringe.pop()

            if current.isWin():
                print(path)
                return path

            current_key = key(current)

            if current_key in closed:



                continue
            else:
                closed.add((current_key, score))

            for successor, action in current.generatePacmanSuccessors():
                successor_score = len(path)+1 + successor.getNumFood()
                fringe.push((successor, path + [action], successor_score), successor_score)

        return path
