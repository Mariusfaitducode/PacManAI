import numpy as np
from pacman_module.game import Agent, Directions
from pacman_module.util import manhattanDistance
from itertools import combinations


def key(state, player):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state:      a game state. See API or class `pacman.GameState`
        player:     boolean, 1 means its max's move, 0 means its min's move

    Returns:
        A hashable key tuple.
    """

    return (state.getPacmanPosition(),
            state.getGhostPosition(1),
            state.getGhostDirection(1),
            state.getFood(),
            tuple(state.getCapsules()),
            player)


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
    """Returns a matrix representing
        the shortest distances between all pairs of empty cells
        and a dictionary mapping each cell to its index in the matrix.

        Arguments:
        state: a game state. See API or class `pacman.GameState`.

        returns:
        - dist:             a numpy array representing the shortest
                            distances between all pairs of empty cells.
        - cell_to_index:    a dictionary mapping each cell to its index
                            in the matrix.
    """
    walls_pos = state.getWalls()
    rows, cols = walls_pos.width, walls_pos.height

    # Want to ONLY keep cells without walls
    open_cells = [(x, y) for x in range(rows) for y in range(cols)
                  if not walls_pos[x][y]]

    # Number of "empty" cells
    nbr_empty_cells = len(open_cells)
    # Dictionary from the mapping from cell to index
    cell_to_index = {cell: index for index, cell in enumerate(open_cells)}

    dist = np.full((nbr_empty_cells, nbr_empty_cells), np.inf)

    # Set the distance from a cell to itself to 0 (all diag elements = 0)
    for id_cell in range(nbr_empty_cells):
        dist[id_cell][id_cell] = 0

    # Check which empty cells are directly adjacent/reachable to each other
    for (x, y) in open_cells:
        current_cell_id = cell_to_index[(x, y)]
        # Check if the neighbors are empty cells
        for x_inc, y_inc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + x_inc, y + y_inc
            if (nx, ny) in cell_to_index:
                neighbor_cell_id = cell_to_index[(nx, ny)]
                # The adjacency matrix is symmetric
                dist[current_cell_id][neighbor_cell_id] = 1
                dist[neighbor_cell_id][current_cell_id] = 1

    # Floyd-Warshall algorithm to compute the shortest real paths between empty
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
    """
    """
    pacman_pos = state.getPacmanPosition()
    # Get the food positions
    food_pos = state.getFood().asList()

    if not food_pos:
        return 0

    # Get the index of the Pacman position in the distance matrix
    pacman_index = cell_to_index[pacman_pos]
    # Get the indices of the food points in the distance matrix
    food_indices = [cell_to_index[food] for food in food_pos
                    if food in cell_to_index]

    food_dist_list = []
    food_pos_list = []
    pos = 0
    for food_index in food_indices:
        food_dist_list.append(dist[pacman_index][food_index])
        food_pos_list.append([food_pos[pos]])
    nearest_food_dist = min(food_dist_list)

    nearest_food_pos = food_pos_list[food_dist_list.index(nearest_food_dist)]
    return nearest_food_dist, nearest_food_pos


def food_score(state, dist, cell_to_index):
    food_pos = state.getFood().asList()

    nearest_food, nearest_food_position = closest_food(
        state, dist, cell_to_index
    )
    food_score = 1 / nearest_food

    # Remove the nearest food
    food_pos.remove(nearest_food_position[0])
    # Get the indices of the food points in the distance matrix
    food_indices = [cell_to_index[food] for food in food_pos
                    if food in cell_to_index]

    for food_index in food_indices:
        food_score += 1 / dist[
            cell_to_index[nearest_food_position[0]]
        ][food_index]
    return food_score


def wall_score(state):
    score = 0
    wall = state.getWalls()
    position = state.getPacmanPosition()
    for w in range(wall.width):
        for h in range(wall.height):
            dist = manhattanDistance((w, h), position)
            if w != 1 or h != 1:
                if wall[w][h]:
                    score += 1/dist
    return score


def ghost_score(state, dist, cell_to_index):
    return 1/ghost_dist(state, dist, cell_to_index)


def closest_food_dist(state, dist, cell_to_index):
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
    pacman_pos = state.getPacmanPosition()
    # Get the food positions
    food_pos = state.getFood().asList()

    if not food_pos:
        return 0

    # Get the index of the Pacman position in the distance matrix
    pacman_index = cell_to_index[pacman_pos]
    # Get the indices of the food points in the distance matrix
    food_indices = [cell_to_index[food] for food in food_pos if
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
    pacman_pos = state.getPacmanPosition()
    # Get the corresponding index in the distance matrix
    pacman_index = cell_to_index[pacman_pos]

    # Get the ghost position
    ghost_pos = state.getGhostPosition(1)
    # Get the corresponding index in the distance matrix
    ghost_index = cell_to_index[ghost_pos]

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
    food_pos = state.getFood().asList()

    # If less than 2 food pts, min_dist = 0
    if len(food_pos) < 2:
        return 0

    # Get the indices of the food points in the distance matrix
    food_indices = [cell_to_index[food] for food in food_pos if
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
                        distance matrix.

        returns:
        max_dist: the maximum distance between any two food cells.
        f1_index: the index of the first food cell in the distance matrix.
        f2_index: the index of the second food cell in the distance matrix.
    """
    food_pos = state.getFood().asList()

    # If less than 2 food pts, min_dist = 0
    if len(food_pos) < 2:
        return 0, 0, 0

    # Get the indices of the food points in the distance matrix
    food_indices = [cell_to_index[food] for food in food_pos if
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
        self.cache = dict()
        self.max_depth = None
        self.floyd_dist = None
        self.floyd_indices = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        # Calculate the floyd distance matrix
        if self.floyd_dist is None:
            self.floyd_dist, self.floyd_indices = floyd_warshall(state)

        # Estimate a max depth value based
        if self.max_depth is None:
            self.max_depth = np.ceil(
                np.log(state.getWalls().width * state.getWalls().height)
                / np.log(4)
            )
            # self.max_depth = 12
        # self.max_depth = max(2, min(5, 2 * state.getNumFood()))
        return self.hminimax(
            state=state,
            player=1,
            depth=0,
            alpha=-np.inf,
            beta=+np.inf,
            _explored=set()
        )[1]

    def cut_off(self, state, depth):
        """Returns whether a cutoff condition is met to stop recursion.

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
            # or ghost_dist(state, self.dist, self.cell_to_index) > 4
            # and depth > 3
        )

    def evaluate(self, state):
        """Returns an estimate of the expected evaluation of the game state.

            Arguments:
            state: a game state. See API or class `pacman.GameState`.

            Returns:
            int: the evaluation score of the state
        """
        if state.isWin():
            return np.inf

        if state.isLose():
            return -np.inf

        pacman_pos = state.getPacmanPosition()
        pacman_index = self.floyd_indices[pacman_pos]

        if state.getNumFood() == 0:
            dist_food_max = 0
            min_dist = 0
        else:
            dist_food_max, f1, f2 = max_inter_food_dist(
                state, self.floyd_dist, self.floyd_indices
            )
            min_dist = min(
                self.floyd_dist[pacman_index][f1],
                self.floyd_dist[pacman_index][f2]
            )

        dist_pac_food_min = closest_food_dist(
            state,
            self.floyd_dist,
            self.floyd_indices
        )

        return (
            state.getScore()
            - 1 * (dist_food_max + min_dist)
            - 2 * dist_pac_food_min
            # + ghost_dist(state, self.floyd_dist, self.floyd_indices) / 2
            - 10 * state.getNumFood()
        )

        # Provide better but less robust results
        # return (
        #     state.getScore()
        #     - 1 * (dist_food_max + min_dist)
        #     - 2 * dist_pac_food_min
        #     - 10 * state.getNumFood() - 1 * wall_score(state)
        #     - 1 * food_score(state, self.floyd_dist, self.floyd_indices)
        #     - 1 * ghost_score(state, self.floyd_dist, self.floyd_indices)
        # )

    def hminimax(self, state, player: bool, depth: int, alpha: float,
                 beta: float, _explored: set):
        """Returns the best possible move using hminimax with alpha-beta pruning.

            Arguments:
            state:     a game state. See API or class `pacman.GameState`.
            player:    boolean, 1 means its max's move, 0 means its min's move
            depth:     integer, current depth of the search tree
            alpha:     float, the best minimum utility score in the current
                       search tree
            beta:      float, the best maximum utility score in the current
                       search tree

            Returns:
            2-tuple containing
                - utility score of the current state
                - legal move as defined in `game.Directions`
        """

        # Move initially returned
        move = Directions.STOP

        # Check cut-off conditions
        if self.cut_off(state, depth):
            return self.evaluate(state), move

        # Check cached states
        current_key = key(state, player)
        if (current_key, state.getScore()) in self.cache:
            return self.cache[(current_key, state.getScore())]

        # Initial best score : -∞ for pacman, +∞ for ghost
        best_score = -np.inf if player else +np.inf

        # Update explored set (copy is necessary: python sucks)
        explored = _explored.copy()
        explored.add(current_key)

        # Determine successors based on player
        successors = (
            state.generatePacmanSuccessors() if player
            else state.generateGhostSuccessors(1)
        )

        # Explore successor nodes
        for successor, action in successors:
            # Check if successor is already explored
            if key(successor, not player) in _explored:
                continue

            # Evaluate next state with of other agent
            eval = self.hminimax(
                successor, not player, depth + 1,
                alpha, beta, explored
            )[0]

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

        self.cache[(current_key, state.getScore())] = (best_score, move)
        return best_score, move
