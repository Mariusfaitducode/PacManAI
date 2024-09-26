from pacman_module.game import Agent, Directions
from pacman_module.util import Stack


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
    """Pacman agent based on depth-first search (DFS)."""

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

        # state_pos, capsules = key(state)

        # if not capsules:
        #     return Directions.STOP

        # target = capsules[0]

        if self.moves is None:
            self.moves = self.bfs(state)

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def bfs(self, state):
        """Given a Pacman game state, returns a list of legal moves to solve
        the search layout.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A list of legal moves.
        """

        path = []
        fringe = Stack()
        fringe.push((state, path))
        closed = set()

        while True:
            if fringe.isEmpty():
                print("Fringe empty", path)
                return path

            current, path = fringe.pop()

            if current.isWin():
                print(path)
                return path

            current_key = key(current)

            if current_key in closed:
                continue
            else:
                closed.add(current_key)

            for successor, action in current.generatePacmanSuccessors():

                print(action)

                fringe.insert((successor, path + [action]))

        return path
