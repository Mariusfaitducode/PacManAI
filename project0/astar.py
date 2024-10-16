from pacman_module.game import Agent, Directions
from pacman_module.util import PriorityQueue
import numpy as np
from functools import lru_cache
from pacman_module.util import manhattanDistance

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
        self.n_capsules = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        if self.n_capsules is None:
            self.n_capsules = len(state.getCapsules())

        if self.moves is None:
            self.moves = self.astar(state)

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def heuristic(self, state):
        """
        Returns the value of the heuristic for a given state.
        Base properties:
        - h(s) >= 0
        - h(s) = 0 for a goal state
        Admissibility property:
        - h(s) <= h*(s), with h*(s) the real cost to reach goal state
        Consistency property:
        - h(s) <= c(s, a, s') + h(s'), with c(s, a, s') the cost of the action a
        - h(s) - 1 <= h(s')
        """
        if state.isWin():
            return 0
        eatenCapsules = self.n_capsules - len(state.getCapsules())

        # Encourage agent to go near capsules
        closeCapsules = 0
        for incY in [-1, 0, 1]:
            for incX in [-1, 0, 1]:
                if state.hasFood(state.getPacmanPosition()[0] + incX, state.getPacmanPosition()[1] + incY):
                    closeCapsules += 1

        return eatenCapsules * 10 + state.getNumFood()

    def score(self, state, path):
        return len(path) + self.heuristic(state)

    def astar(self, state):
        path = []
        fringe = PriorityQueue()
        score = self.score(state, path)
        fringe.push((state, path), score)
        closed = set()

        while True:
            if fringe.isEmpty():
                return path

            priority, (current, path) = fringe.pop()

            if current.isWin():
                return path

            current_key = key(current)

            if current_key in closed:
                continue
            else:
                closed.add(current_key)

            for successor, action in current.generatePacmanSuccessors():
                new_path = path + [action]
                successor_score = self.score(
                    successor, new_path,
                )
                fringe.push((successor, new_path), successor_score)
        return path
