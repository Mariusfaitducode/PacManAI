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


def get_extreme_foods(state):
    food_positions = state.getFood()
    pacman_position = state.getPacmanPosition()

    # Store the positions of extreme the dots
    # 0 : extreme left, 1 : extreme right, 2 : extreme bottom, 3 : extreme top.
    extreme_foods = [pacman_position for _ in range(4)]
    for i in range(food_positions.width):
        for j in range(food_positions.height):
            if not food_positions[i][j]:
                continue
            if i < extreme_foods[0][0]:
                extreme_foods[0] = (i, j)
            if i > extreme_foods[1][0]:
                extreme_foods[1] = (i, j)
            if j < extreme_foods[2][1]:
                extreme_foods[2] = (i, j)
            if j > extreme_foods[3][1]:
                extreme_foods[3] = (i, j)
    return extreme_foods


def floyd_warshall(state):
    """Given a Pacman game state, return a matrix representing
        the shortest distances between all pairs of empty cells
        and a dictionary mapping each cell to its index in the matrix.

        Arguments:
        state: a game state. See API or class `pacman.GameState`.

        returns:
        - dist:             a numpy array representing the shortest
                            distances between all pairs of empty cells.
        - cell_to_index:    a dictionary mapping each cell to itsindex
                            in the matrix.
    """
    walls_Pos = state.getWalls()
    rows, cols = walls_Pos.width, walls_Pos.height

    # Want to ONLY keep cells without walls
    open_cells = [(x, y) for x in range(rows) for y in range(cols)
                  if not walls_Pos[x][y]]

    # Number of "empty" cells
    nbr_empty_cells = len(open_cells)
    # Dictionnary from the mapping from cell to index
    cell_to_index = {cell: id for id, cell in enumerate(open_cells)}

    dist = np.full((nbr_empty_cells, nbr_empty_cells), np.inf)

    # Set the distance from a cell to itself to 0 (all diag elements = 0)
    for id_cell in range(nbr_empty_cells):
        dist[id_cell][id_cell] = 0

    # Check which empty cells are directly adjacent/reachable to each other
    for (x, y) in open_cells:
        current_cell_id = cell_to_index[(x, y)]
        # Check if the neighbors are empty cells
        for x_neighb, y_neighb in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + x_neighb, y + y_neighb
            if (nx, ny) in cell_to_index:
                neighbor_cell_id = cell_to_index[(nx, ny)]
                # The adjacency matrix is symmetric
                dist[current_cell_id][neighbor_cell_id] = 1
                dist[neighbor_cell_id][current_cell_id] = 1

    # Floyd-Warshall algorithm to compute shortest real paths between empty
    for k in range(nbr_empty_cells):
        for i in range(nbr_empty_cells):
            for j in range(nbr_empty_cells):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist, cell_to_index


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
        extreme_foods[i], extreme_foods[max_index] =\
            extreme_foods[max_index], extreme_foods[i]
        current_pos = extreme_foods[i]
        heuristic += min(dist)

    return heuristic


def closest_food(state, dist, cell_to_index):
    # Get pacman position
    pacman_Pos = state.getPacmanPosition()
    # Get the food positions
    food_Pos = state.getFood().asList()

    if not food_Pos:
        return 0

    # Get the index of the Pacman position in the distance matrix
    pacman_index = cell_to_index[pacman_Pos]
    # Get the indices of the food points in the distance matrix
    food_indices = [cell_to_index[food] for food in food_Pos
                    if food in cell_to_index]

    food_dist_list = []
    food_pos_list = []
    pos = 0
    for food_index in food_indices:
        food_dist_list.append(dist[pacman_index][food_index])
        food_pos_list.append([food_Pos[pos]])
    nearest_food_dist = min(food_dist_list)

    nearest_food_Pos = food_pos_list[food_dist_list.index(nearest_food_dist)]
    return nearest_food_dist, nearest_food_Pos


def food_score(state, dist, cell_to_index):
    food_Pos = state.getFood().asList()

    nearestFood, nearestFoodPosition = closest_food(state, dist, cell_to_index)
    foodScore = 1 / nearestFood

    # Remove the nearest food
    food_Pos.remove(nearestFoodPosition[0])
    # Get the indices of the food points in the distance matrix
    food_indices = [cell_to_index[food] for food in food_Pos
                    if food in cell_to_index]

    for food_index in food_indices:
        foodScore += 1/dist[cell_to_index[nearestFoodPosition[0]]][food_index]
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


def ghost_score(state, dist, cell_to_index):
    return 1/ghost_dist(state, dist, cell_to_index)


def closest_food_Pac_dist(state, dist, cell_to_index):
    """Given a Pacman game state a matrix representing the distance
        between each pair of empty cells, and a dictionary mapping each
        returns the distance between Pacman and the closest food.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.
        dist: a numpy array representing the shortest distances between
                all pairs of empty cells.
        cell_to_index: a dictionary mapping each cell to its index in the
                        dist matrix.

    Returns:
        A int.
    """
    # Get pacman position
    pacman_Pos = state.getPacmanPosition()
    # Get the food positions
    food_Pos = state.getFood().asList()

    if not food_Pos:
        return 0

    # Get the index of the Pacman position in the distance matrix
    pacman_index = cell_to_index[pacman_Pos]
    # Get the indices of the food points in the distance matrix
    food_indices = [cell_to_index[food] for food in food_Pos if
                    food in cell_to_index]

    return min(dist[pacman_index][food_index] for food_index
               in food_indices)


def ghost_dist(state, dist, cell_to_index):
    """Given a Pacman game state a matrix representing the distance
        between each pair of empty cells, and a dictionary mapping each
        returns the distance between Pacman and the ghost.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.
        dist: a numpy array representing the shortest distances between
                all pairs of empty cells.
        cell_to_index: a dictionary mapping each cell to its index in the
                        dist matrix.

    Returns:
        A int.
    """
    # Get Pacman position
    pacman_Pos = state.getPacmanPosition()
    # Get the corresponding index in the distance matrix
    pacman_index = cell_to_index[pacman_Pos]

    # Get the ghost position
    ghost_Pos = state.getGhostPosition(1)
    # Get the corresponding index in the distance matrix
    ghost_index = cell_to_index[ghost_Pos]

    return dist[pacman_index][ghost_index]


def min_inter_food_dist(state, dist, cell_to_index):
    """ Given a Pacman game state a matrix representing the distance
        between each pair of empty cells, and a dictionary mapping each
        returns the distance between the two furthest food cells and
        their corresponding indexes in the distance matrix.

        Arguments:
        state: a game state. See API or class `pacman.GameState`.
        dist: a numpy array representing the shortest distances between
                all pairs of empty cells.
        cell_to_index: a dictionary mapping each cell to its index in the
                        dist matrix.

        Returns:
        max_dist: the maximum distance between any two food cells.
        f1_index: the index of the first food cell in the distance matrix.
        f2_index: the index of the second food cell in the distance matrix.
    """
    food_Pos = state.getFood().asList()

    # If less than 2 food pts, min_dist = 0
    if len(food_Pos) < 2:
        return 0

    # Get the indices of the food points in the distance matrix
    food_indices = [cell_to_index[food] for food in food_Pos if
                    food in cell_to_index]

    # Find the minimum distance between any two foods
    min_dist = np.inf
    for f1, f2 in combinations(food_indices, 2):
        distance = dist[f1][f2]
        if distance < min_dist:
            min_distance = distance

    return min_distance


def max_inter_food_dist(state, dist, cell_to_index):
    """ Given a Pacman game state a matrix representing the distance
        between each pair of empty cells, and a dictionary mapping each
        returns the distance between the two furthest food cells and
        their corresponding indexes in the distance matrix.

        Arguments:
        state: a game state. See API or class `pacman.GameState`.
        dist: a numpy array representing the shortest distances between
                all pairs of empty cells.
        cell_to_index: a dictionary mapping each cell to its index in the
                        dsit matrix.

        returns:
        max_dist: the maximum distance between any two food cells.
        f1_index: the index of the first food cell in the distance matrix.
        f2_index: the index of the second food cell in the distance matrix.
    """
    food_Pos = state.getFood().asList()

    # If less than 2 food pts, min_dist = 0
    if len(food_Pos) < 2:
        return 0, 0, 0

    # Get the indices of the food points in the distance matrix
    food_indices = [cell_to_index[food] for food in food_Pos if
                    food in cell_to_index]

    max_dist = - np.inf
    f1_index = -1
    f2_index = -1

    # Find the maximum distance between any two foods
    for f1, f2 in combinations(food_indices, 2):
        distance = dist[f1][f2]
        if distance > max_dist:
            f1_index = f1
            f2_index = f2
            max_distance = distance

    # Return the maximum distance and the index of the corresponding
    # food points
    return max_distance, f1_index, f2_index


class PacmanAgent(Agent):
    """Pacman agent with enhanced loop avoidance based on minimax."""

    def __init__(self):
        super().__init__()
        self.key_map = {}
        self.max_depth = None
        self.dist = None
        self.cell_to_index = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        if self.dist is None:
            self.dist, self.cell_to_index = floyd_warshall(state)

        self.max_depth = max(2, min(5, 2 * state.getNumFood()))

        return self.hminimax(
            state=state,
            player=1,
            depth=0,
            alpha=-np.inf,
            beta=+np.inf)[1]

    def cutOff(self, state, depth):
        """Evaluate if one of the cutoff conditions is met to stop recursion.

        Arguments:
            state:  a game state. See API or class `pacman.GameState`.
            depth:  integer, current depth of the search tree

        Returns:
            A boolean value.
        """
        return (
            state.isWin()
            or state.isLose()
            or depth > self.max_depth
            or ghost_dist(state, self.dist, self.cell_to_index) > 4
            and depth > 3
        )

    def eval(self, state):
        """Given a Pacman game state, returns an estimate of the
            expected utility of the game state.

            Arguments:
            state: a game state. See API or class `pacman.GameState`.

            Returns:
            A int.
        """
        if state.isWin():
            return 5000 + state.getScore()

        if state.isLose():
            return -5000 + state.getScore()

        pacman_Pos = state.getPacmanPosition()
        pacman_index = self.cell_to_index[pacman_Pos]

        if (len(state.getFood().asList()) == 0):
            dist_food_max = 0
            min_dist = 0
        else:
            dist_food_max, f1, f2 =\
                max_inter_food_dist(state, self.dist, self.cell_to_index)
            min_dist = min(self.dist[pacman_index][f1],
                           self.dist[pacman_index][f2])

        dist_pac_food_min =\
            closest_food_Pac_dist(state, self.dist, self.cell_to_index)

        return state.getScore() - 1 * (dist_food_max + min_dist)\
            - 2 * dist_pac_food_min - 10 * state.getNumFood()

        # Provide sometimes better results but les robust
        # return state.getScore() -  1*(dist_food_max + min_dist) - 2*dist_pac_food_min \
        # - 10 * state.getNumFood() - 1 * wall_score(state) - 1 * \
        #     food_score(state, self.dist, self.cell_to_index) - 1 *\
        #             ghost_score(state, self.dist, self.cell_to_index)

    def hminimax(self, state, player: bool, depth: int, alpha: float,
                 beta: float):
        """Given a Pacman game state, returns the best possible move
            using hminimax with alpha-beta pruning.

            Arguments:
            state:     a game state. See API or class `pacman.GameState`.
            player:    boolean, 1 means its max's move, 0 means its min's move
            depth:     integer, current depth of the search tree
            alpha:     float, the best minimum utility score in the current
                       search tree
            beta:      float, the best maximum utility score in the current
                       search tree

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
            eval = self.hminimax(successor, not player, depth + 1,
                                 alpha, beta)[0]

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

        return best_score, move
