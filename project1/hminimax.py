import numpy as np

from pacman_module.game import Agent, Directions
from pacman_module.util import manhattanDistance
from itertools import combinations

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

def get_extreme_foods(state):
    food_positions = state.getFood()
    pacman_position = state.getPacmanPosition()

    # Store the positions of extreme the dots
    # 0 : extreme left, 1 : extreme right, 2 : extreme bottom, 3 : extreme top.
    extreme_foods = [pacman_position for _ in range(4)]
    for i in range(food_positions.width):
        for j in range(food_positions.height):
            if not food_positions[i][j]: continue
            if i < extreme_foods[0][0]:
                extreme_foods[0] = (i, j)
            if i > extreme_foods[1][0]:
                extreme_foods[1] = (i, j)
            if j < extreme_foods[2][1]:
                extreme_foods[2] = (i, j)
            if j > extreme_foods[3][1]:
                extreme_foods[3] = (i, j)
    return extreme_foods

def floyd_marshall_distance(state):
    extreme_foods = get_extreme_foods(state)

    # Floyd distance of extreme foods
    heuristic = 0
    current_pos = state.getPacmanPosition()
    for i in [3, 2, 1, 0]:
        dist = [
            manhattanDistance(current_pos, extreme_foods[j])
            for j in {0, i}
        ]
        max_index = dist.index(max(dist))
        extreme_foods[i], extreme_foods[max_index] = extreme_foods[max_index], extreme_foods[i]
        current_pos = extreme_foods[i]
        heuristic += min(dist)

    return heuristic

def closest_food(state):
    foods = state.getFood().asList()
    pacmanPosition = state.getPacmanPosition()
    foodDistances = []
    foodPosition = []
    for food in foods:
        foodDistances.append(manhattanDistance(pacmanPosition, food))
        foodPosition.append([food])
    nearestFood = min(foodDistances)

    nearestFoodPosition = foodPosition[foodDistances.index(nearestFood)]
    return nearestFood, nearestFoodPosition

def food_score(state):
    foods = state.getFood().asList()

    nearestFood, nearestFoodPosition = closest_food(state)
    foodScore = 1 / nearestFood
    foods.remove(nearestFoodPosition[0])
    for food in foods:
        foodScore += 1/manhattanDistance(
            food, nearestFoodPosition[0]
        )
    return foodScore

def wall_score(state):
    score = 0
    wall = state.getWalls()
    W = wall.width
    H = wall.height
    position = state.getPacmanPosition()
    for w in range(W):
        for h in range(H):
            dist = manhattanDistance((w, h), position)
            if w != 1 or h != 1:
                if wall[w][h]:
                    score += 1/dist
    return score

def ghost_score(state):
    return 1/manhattanDistance(
        state.getPacmanPosition(),
        state.getGhostPosition(1)
    )

def pierre_heuristic(state, node, fw):
    """Given a Pacman game state, calculate x as the real distance between
        the two furthest foods from each other (f1 and f2). And y, the
        minimum of the real distance between pacman and one those
        foods (f1 or f2). The real distance is calculated taking account
        the geometry of the maze.
    """

    pos = state.getPacmanPosition()
    food = state.getFood().asList()

    if len(food) == 0:
        return 0

    pacman_ind = node.get(pos)
    food_ind = [node.get(i) for i in food]

    if len(food) == 1:
        return fw[pacman_ind][food_ind[0]]

    x, p1, p2 = -1, -1, -1
    for i in food_ind:
        for j in food_ind:
            if fw[i][j] > x:
                x = fw[i][j]
                p1, p2 = i, j

    y = min(fw[pacman_ind][p1], fw[pacman_ind][p2])

    return x + y

def closest_food_dist(state):
    pacman = state.getPacmanPosition()
    food = state.getFood().asList()

    if not food: return 0

    return min(
        manhattanDistance(pacman, food)
        for food in food
    )

def ghost_dist(state):
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

def min_inter_food_dist(state):
    """
    Return the minimal distance between all foods in the state
    """
    foodPos = state.getFood().asList()

    if len(foodPos) < 2: return 0

    return min(
        manhattanDistance(f1, f2) for f1, f2
        in combinations(foodPos, 2)
    )

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
            beta=+np.inf
        )[1]

    def cutOff(self, state, depth):
        """Evaluate if one of the cutoff conditions is met to stop recursion.

        Arguments:
            state:          a game state. See API or class `pacman.GameState`.
            depth:          integer, current depth of the search tree

        Returns:
            A boolean value.
        """
        return (
            state.isWin()
            or state.isLose()
            or depth > self.max_depth
            or ghost_dist(state) > 4 and depth > 3
        )

    def eval(self, state):
        # -------- General end state score --------
        if state.isWin(): return 5000 + state.getScore()
        if state.isLose(): return -5000 + state.getScore()

        # -------- Our score --------
        # if nbfoodinit > state.getNumFood():
        #     if self.max_depth > depth:
        #         self.depth_max = depth
        #     return (
        #             state.getScore() - depth
        #             - 10 * state.getNumFood()
        #     )

        return (
            state.getScore()
            - (3.5 if state.getNumFood() <= 1 else 0)  * closest_food_dist(state)
            - (2 if ghost_dist(state) > 5 else 1) * closest_food_dist(state)
            - (3 if state.getNumFood() <= 1 else 1) * 10 * state.getNumFood()
        )

        # -------- Reno's score --------
        # return (
        #     2       * foodScore(state)
        #     + .1    * wallScore(state)
        #     - 2     * ghostScore(state)
        # )

        # -------- Pierre's buddy score --------
        # return (
        #     state.getScore()
        #     # REAL dist between two furthest foods + min real dist between pacman and these foods
        #     - pierre_heuristic(state, self.node, self.FW)
        #     # REAL dist between pacman and closest food
        #     - 3.5 * min_food_dist(state)
        #     - 10 * state.getNumFood()
        # )

        # -------- Hugo's score --------
        # return state.getScore() + floyd_marshall_distance(state)

    def hminimax(self, state, player: bool, depth: int, alpha: float, beta: float):
        """Given a Pacman game state, returns the best possible move
            using hminimax with alpha-beta pruning.

            Arguments:
            state:      a game state. See API or class `pacman.GameState`.
            player:     boolean, 1 means its max's move, 0 means its min's move
            depth:      integer, current depth of the search tree
            alpha:      float, the best minimum utility score in the current search tree
            beta:       float, the best maximum utility score in the current search tree

            Returns:
            tuple of 4 elements
                - utility score of the current state
                - legal move as defined in `game.Directions`
        """

        # Move initially returned
        move = Directions.STOP

        # Check cut-off conditions
        if self.cutOff(state, depth):
            return self.eval(state), move

        # Check cached states
        current_key = key(state, player)
        if current_key in self.key_map:
            inDepth = self.key_map[current_key]

            if inDepth > depth:
                self.key_map[current_key] = depth
            else:
                if player:
                    return 5000, Directions.STOP
                return -5000, Directions.STOP
        else:
            self.key_map[current_key] = depth

        # Initial best score : -∞ for pacman, +∞ for ghost
        best_score = -np.inf if player else +np.inf

        # Determine successors based on player
        successors = (
            state.generatePacmanSuccessors() if player
            else state.generateGhostSuccessors(1)
        )

        # Explore successor nodes
        for successor, action in successors:
            # Evaluate next state with of other agent
            eval = self.hminimax(successor, not player, depth + 1, alpha, beta)[0]

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