import numpy as np

from pacman_module.game import Agent, Directions
from pacman_module.util import manhattanDistance
from itertools import combinations

"""
+-----------------+-----------------+-----------------+-----------------+
|  Scores         |  Small          |  Medium         |  Large          |
+-----------------+-----------------+-----------------+-----------------+
|  Dumby          |      516        |      539        |      536        |
+-----------------+-----------------+-----------------+-----------------+
|  Smarty         |      516        |      539        |      534        |
+-----------------+-----------------+-----------------+-----------------+
|  Greedy         |      516        |      539        |      534        |
+-----------------+-----------------+-----------------+-----------------+


+-----------------+-----------------+-----------------+-----------------+
|  Nbr of nodes   |  Small          |  Medium         |  Large          |
+-----------------+-----------------+-----------------+-----------------+
|  Dumby          |      11         |      248        |      654        |
+-----------------+-----------------+-----------------+-----------------+
|  Smarty         |      11         |      218        |      722        |
+-----------------+-----------------+-----------------+-----------------+
|  Greedy         |      11         |      221        |      722        |
+-----------------+-----------------+-----------------+-----------------+
"""


def key(state, maxPlayer):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state:      a game state. See API or class `pacman.GameState`
        maxPlayer:  boolean, 1 means its max's move, 0 means its min's move

    Returns:
        A hashable key tuple.
    """

    return (state.getPacmanPosition(),
            state.getGhostPosition(1),
            state.getGhostDirection(1),
            state.getFood(),
            tuple(state.getCapsules()),
            maxPlayer)


class PacmanAgent(Agent):
    """Pacman agent with enhanced loop avoidance based on minimax."""

    def __init__(self):
        super().__init__()
        self.key_map = {}
        self.max_depth = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        self.max_depth = max(2, min(4, 2 * state.getNumFood()))

        return self.hminimax(
            state=state,
            player=1,
            depth=0,
            alpha=-np.inf,
            beta=+np.inf,
            nbfoodinit=state.getNumFood()
        )[1]

    def cutOff(self, state, depth, nbfoodinit):
        """Evaluate if one of the cutoff conditions is met to stop recursion.

        Arguments:
            state:          a game state. See API or class `pacman.GameState`.
            depth:          integer, current depth of the search tree
            nbfoodinit:     integer, number of food dots at the beginning
                            of the game

        Returns:
            A boolean value.
        """

        if (state.isWin() or state.isLose() or depth > self.max_depth or
                nbfoodinit > state.getNumFood()):
            return True

        elif self.getPacmanGhostdist(state) > 4 and depth > 3:
            return True

        else:
            return False

    def getPacmanGhostdist(self, state):
        """Given a Pacman game state, returns the distance between Pacman
        and the ghost.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A int.
        """
        pacmanPos = state.getPacmanPosition()
        ghostPos = state.getGhostPosition(1)

        return manhattanDistance(pacmanPos, ghostPos)

    def getClosestFoodDist(self, state):
        """Given a Pacman game state, returns the distance between Pacman
        and the closest food dot.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A int.
        """
        pacmanPos = state.getPacmanPosition()
        foodPos = state.getFood().asList()

        if not foodPos:
            return 0

        closestFoodDist = min(manhattanDistance(pacmanPos, food)
                              for food in foodPos)

        return closestFoodDist

    def getMinFoodDist(self, state):
        """Given a Pacman game state, returns the minimum distance between
        two food dots.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A int.
        """
        foodPos = state.getFood().asList()

        if len(foodPos) < 2:
            return 0

        minFoodDist = min(manhattanDistance(f1, f2) for f1, f2
                          in combinations(foodPos, 2))

        return minFoodDist

    def eval(self, state, depth, nbfoodinit):
        """Given a Pacman game state, returns an estimate of the
            expected utility of the game state.

            Arguments:
            state:          a game state. See API or class `pacman.GameState`.
            depth:          integer, current depth of the search tree
            nbfoodinit:     integer, number of food dots at the beginning of
                            the game

            Returns:
            A int.
        """

        if state.isWin():
            return 5000 + state.getScore()

        if state.isLose():
            return -5000 + state.getScore()

        if nbfoodinit > state.getNumFood():
            if self.max_depth > depth:
                self.depth_max = depth
            return (
                    state.getScore() - depth
                    - 10 * state.getNumFood()
            )

        distFoodMin = self.getMinFoodDist(state)
        distPacFoodMin = self.getClosestFoodDist(state)
        dist = self.getPacmanGhostdist(state)

        if state.getNumFood() <= 1:
            return (state.getScore()) - distPacFoodMin\
                    - 3 * 10 * state.getNumFood()

        elif dist > 5:
            return state.getScore() - 3.5 * distFoodMin - distPacFoodMin\
                    - 10 * state.getNumFood()

        else:
            return state.getScore() - 3.5 * distFoodMin - 2 * distPacFoodMin\
                    - 10 * state.getNumFood()

    def hminimax(self, state, player: bool, depth: int, alpha: float, beta: float, nbfoodinit: int):
        """Given a Pacman game state, returns the best possible move
            using hminimax with alpha-beta pruning.

            Arguments:
            state:      a game state. See API or class `pacman.GameState`.
            player:     boolean, 1 means its max's move, 0 means its min's move
            depth:      integer, current depth of the search tree
            alpha:      float, the best minimum utility score in the current search tree
            beta:       float, the best maximum utility score in the current search tree
            nbfoodinit: integer, number of food dots at the beginning of the game

            Returns:
            tuple of 4 elements
                - utility score of the current state
                - legal move as defined in `game.Directions`
        """

        # Move initially returned
        move = Directions.STOP

        # Check cut-off conditions
        if self.cutOff(state, depth, nbfoodinit):
            return self.eval(state, depth, nbfoodinit), move

        # Initial best score : -∞ for pacman, +∞ for ghost
        best_score = -np.inf if player else +np.inf

        currentKey = key(state, player)

        if currentKey in self.key_map:
            inDepth = self.key_map[currentKey]

            if inDepth > depth:
                self.key_map[currentKey] = depth
            else:
                if player:
                    return 5000, Directions.STOP
                return -5000, Directions.STOP
        else:
            self.key_map[currentKey] = depth

        # Determine successors based on player
        successors = (
            state.generatePacmanSuccessors() if player
            else state.generateGhostSuccessors(1)
        )

        # Explore successor nodes
        for successor, action in successors:
            eval = self.hminimax(successor, not player, depth + 1, alpha, beta, nbfoodinit)[0]

            # Max player (pacman)
            if player:
                alpha_pruning = eval >= beta
                better_score = eval > best_score

                # Best score / best move update
                if alpha_pruning or better_score:
                    best_score = eval
                    move = action

                # Alpha pruning
                if alpha_pruning: break
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
                if beta_pruning: break
                beta = min(beta, eval)

        return best_score, move