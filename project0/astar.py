from pacman_module.game import Agent, Directions
from pacman_module.util import PriorityQueue


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

    def heuristic(self, state, previous_capsules):
        """Returns the value of the heuristic for a given state."""
        if state.getPacmanPosition() in previous_capsules:
            return 100
        return state.getNumFood()

    def score(self, state, path, previous_capsules):
        return len(path) + self.heuristic(state, previous_capsules)

    def astar(self, state):
        path = []
        fringe = PriorityQueue()
        score = self.score(state, path, state.getCapsules())
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
                    current.getCapsules()
                )
                fringe.push((successor, new_path), successor_score)

        return path
