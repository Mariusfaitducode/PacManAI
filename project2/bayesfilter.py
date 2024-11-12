import numpy as np
from scipy.stats import binom

from pacman_module.game import Agent, Directions, manhattanDistance
from pacman_module.util import Queue, PriorityQueue


"""
Bayes filter with BFS
Layout               Ghost     #Ghost   Score    Time [s]     Error
-------------------  -------   ------   --------  ----------  -------
large_filter         fearless  1             676       4.520       --
large_filter         afraid    1             648       4.797       --
large_filter         terrified 1             644       5.039       --
-------------------  -------   ------   --------  ----------  -------
large_filter_walls   fearless  1             650       9.125       --
large_filter_walls   afraid    1             654       7.793       --
large_filter_walls   terrified 1             426      12.378       --
-------------------  -------   ------   --------  ----------  -------
zones                fearless  4            1263       4.383       --
zones                afraid    4            1179       6.670       --
zones                terrified 4            1251       4.569       --
-------------------  -------   ------   --------  ----------  -------
grid                 fearless  4            1263       4.383       --
grid                 afraid    4            1179       6.670       --
grid                 terrified 4            1251       4.569       --
-------------------  -------   ------   --------  ----------  -------
"""


"""
Bayes filter with A*, floyd distance heuristic
Layout               Ghost     #Ghost   Score    Time [s]        Error
-------------------  -------   ------   --------  -------------  -------
large_filter         fearless  1             676   4.724 (+0.2)    --
large_filter         afraid    1             648   4.797 (-0.1)    --
large_filter         terrified 1             644   5.668 (+0.6)    --
-------------------  -------   ------   --------  -------------  -------
large_filter_walls   fearless  1             650   3.182 (-7.8)    --
large_filter_walls   afraid    1             654   3.571 (-4.1)    --
large_filter_walls   terrified 1             426   5.285 (-6.7)   --
-------------------  -------   ------   --------  -------------  -------
zones                fearless  4            1263   2.108 (-2.6)    --
zones                afraid    4            1179   2.575 (-3.2)    --
zones                terrified 4            1251   1.908 (-2.6)    --
-------------------  -------   ------   --------  -------------  -------
grid                 fearless  4            1263   1.265 (-3.1)     --
grid                 afraid    4            1179   1.322 (-4.7)     --
grid                 terrified 4            1251   1.047 (-3.5)     --
-------------------  -------   ------   --------  -------------  -------
"""

def floyd_warshall(walls):
    """Given a Pacman game state, return a matrix representing
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
    walls_pos = walls
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


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        # The type of ghost: afraid, fearless or terrified
        self.ghost = ghost

        # Dictionary associating the ghost type to a specific level of fear
        # The more affraid the ghost is, the larger the value will be
        self.ghost_dict = {
            "afraid": 1.0,
            "fearless": 0.0,
            "terrified": 3.0,
        }

        # Binomial number of trials
        self.n = 4
        # Binomial probability of success
        self.p = 0.5

    def transition_matrix(self, walls, position):
        """Builds the transition matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (k, l) | X_{t-1} = (i, j)) for
            the ghost to move from (i, j) to (k, l).
        """

        W = walls.width
        H = walls.height

        # Possible moves for a cell (going [West, East, North, South])
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Defintion of the transition matrix with the provided dimensions
        T_matrix = np.zeros((W, H, W, H))

        # Probability of the ghost to be affraid is 2^fear_level
        ghost_fear_prob = 2**self.ghost_dict[self.ghost]

        # Loop over each cell of the layout
        for i in range(W):
            for j in range(H):
                # To store the probability of the ghost to be in
                # the 4 possible cells
                prob_ghost = np.zeros(4)
                counter = 0

                # If the cell is a wall, prob = 0
                if walls[i][j]:
                    continue

                # Current distance between pacman and the ghost
                pacm_ghost_dist = manhattanDistance((i, j), position)

                # The probability of the ghost to be in another cell
                # Provided the 4 possible moves ([right, left, up, down])
                for (dx, dy) in moves:
                    x = i + dx
                    y = j + dy

                    # Verify if the index is not out of the range
                    # And that it is NOT a wall
                    if (x < 0 or x >= W or y < 0 or y >= H or walls[x][y]):
                        counter += 1
                        continue

                    # New distance between pacman and the "new" ghost position
                    pacm_ghost_dist_new = manhattanDistance((x, y), position)

                    # If the ghost is afraid, the probability of the ghost
                    # to be further is 2^fear_level otherwise 1
                    if pacm_ghost_dist < pacm_ghost_dist_new:
                        prob_ghost[counter] = ghost_fear_prob
                    else:
                        prob_ghost[counter] = 1

                    counter += 1

                # Normalization of the probabilities
                norm_term = np.sum(prob_ghost)
                if norm_term != 0:
                    prob_ghost = prob_ghost / norm_term

                counter = 0
                # Update of the transition matrix
                for (dx, dy) in moves:
                    x = i + dx
                    y = j + dy

                    # Verify if the index is not out of the range
                    # And that it is NOT a wall
                    if (x < 0 or x >= W or y < 0 or y >= H or walls[x][y]):
                        counter += 1
                        continue
                    else:
                        T_matrix[i, j, x, y] = prob_ghost[counter]

                    counter += 1

        # Return the transition matrix
        return T_matrix

    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """
        # Dimensions of the layout
        W = walls.width
        H = walls.height

        # Definition of the observation matrix with the provided dimensions
        O_matrix = np.zeros((W, H))

        mean_bin = self.n*self.p

        # Loop over each cell of the layout
        for i in range(W):
            for j in range(H):
                # If the cell is a wall, the ghost CANNOT be there
                if walls[i][j]:
                    continue

                # Distance between pacman and the supposed ghost position
                dist_pac_cell = manhattanDistance(position, (i, j))

                # Extraction of the variable following the binomial
                z = evidence - dist_pac_cell + mean_bin

                # Only z only in the range [0, n]
                if (0 <= z <= self.n):
                    O_matrix[i, j] = binom.pmf(z, self.n, self.p)

        # Return the observation matrix
        return O_matrix

    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """
        # Dimensions of the layout
        W = walls.width
        H = walls.height

        # Possible moves for a cell (going [West, East, North, South])
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Definition of the updated belief state with the provided dimensions
        updated_belief = np.zeros((W, H))

        # Transition matrix
        T_matrix = self.transition_matrix(walls, position)
        # Observation matrix
        O_matrix = self.observation_matrix(walls, evidence, position)

        # Loop over each cell of the layout
        for i in range(W):
            for j in range(H):
                # Account for the product of the possible transition
                # and previous belief
                transition_belief = 0

                # If the cell is a wall, the ghost CANNOT be there
                if walls[i][j]:
                    continue

                # Loop over the possible moves
                # ∑ P(X_t = (i, j) | X_{t-1} = (x, y)) * b_{t-1}(x, y)
                for (dx, dy) in moves:
                    x = i + dx
                    y = j + dy

                    # Verify if the index is not out of the range
                    # And that it is NOT a wall
                    if (x < 0 or x >= W or y < 0 or y >= H or walls[x][y]):
                        continue

                    # Compute product of the transition matrix
                    # and the previous belief
                    transition_belief += T_matrix[i, j, x, y] * belief[x, y]

                # Update of the belief state
                updated_belief[i, j] = O_matrix[i, j] * transition_belief

        # Normalization of the belief state to get valid probabilities
        updated_belief /= updated_belief.sum()

        # Return the updated belief state
        return updated_belief

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls,
                    beliefs[i],
                    evidences[i],
                    position,
                )

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()
        # Matrix of shortest distances between any pairs of empty cells
        self.dist = None
        # Dictionary mapping each cell to its index in the matrix
        self.cell_to_index = None
        # Index of the targetted ghost
        self.targetted_ghosts = None
        # The position of the targetted ghost
        self.targetted_ghosts_pos = None
        # Remaining attention time to the actual targetted ghost
        # while another ghost has be detected closer
        self.attention_durations = None
        # Possible moves
        self.moves = [
            (-1, 0, Directions.WEST),
            (1, 0, Directions.EAST),
            (0, -1, Directions.SOUTH),
            (0, 1, Directions.NORTH)
        ]

    def closes_ghost_positions(self, beliefs, eaten, position):
        """Provide the index of the closest ghost for which the maximum
        position belief provide the closes distance to pacman's position.

        Arguments:
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A int representing the ghost identification,
            a int representing a distance.
        """
        # Total nbr of ghosts
        nbr_ghosts = len(eaten)
        # To keep the most probable ghost position
        ghost_pos = np.zeros((nbr_ghosts, 2))
        # To ket the distance between pacman and each ghost
        # most probable position
        dist_pac_ghost = np.zeros(nbr_ghosts)

        # Loop over each ghost
        for i in range(nbr_ghosts):
            if eaten[i]:
                dist_pac_ghost[i] = np.inf
                continue

            # Value of the most probable position
            ghost_pos_belief = 0

            # print("Belief", beliefs[i])
            # Loop over the current ghost belief states
            # Want to look for the most probable ghost position
            # and save the belief and the position
            for (j, k), prob in np.ndenumerate(beliefs[i]):
                # If this position have a larger belief than previous ones
                if prob > ghost_pos_belief:
                    # Save the prob and the position
                    ghost_pos_belief = prob
                    ghost_pos[i] = (j, k)

            # Compute the real distance between pacman
            # and that most probable position
            dist_pac_ghost[i] = \
                self.dist[
                    self.cell_to_index[position],
                    self.cell_to_index[tuple(ghost_pos[i])]
                ]

        # Index of the closest ghost
        closest_ghost = np.argmin(dist_pac_ghost)

        # Return the closest ghost index for the identification
        # and its position
        return closest_ghost, ghost_pos[closest_ghost]

    def update_targetted_ghost(self, closest_gost, ghost_pos):
        """Given the closest ghost at the current step, update the ghost
        targetted by pacman if a ghost has been closer to pacman for 4
        following steps than its actual targetted.

        Arguments:
            closest_gost: The index of the closest ghost to pacman.
            ghost_pos: The position the closest ghost.

        Returns:
            None
        """
        # Identify the first closest ghost
        if self.targetted_ghosts is None:
            self.targetted_ghosts = closest_gost
            self.targetted_ghosts_pos = ghost_pos
            self.attention_durations = 4

        # If the closest ghost is not the targetted one
        if self.targetted_ghosts != closest_gost:
            # If the attention duration is over, change of targetted_ghosts
            if self.attention_durations == 0:
                self.targetted_ghosts = closest_gost
                self.attention_durations = 4
            # If the attention duration is not over,
            # decrease the remaining attention durantion for that ghost
            else:
                self.attention_durations -= 1
        else:
            # If the closest ghost is the targetted one,
            # reset the attention duration
            self.attention_durations = 4

    def astar(self, pacman_pos:tuple, ghost_pos:np.ndarray, walls):
        """Given Pacman's position, a ghost estimated position and the layout,
        returns a list of legal moves to reach the ghost computed using A* with Floyd distances.

        Arguments:
            pacman_pos: The starting position (x, y) of Pacman
            ghost_pos: The target position (x, y) of the ghost
            walls: The W x H grid of the layout's walls

        Returns:
            A list of legal moves.
        """

        fringe = PriorityQueue()
        path = []
        fringe.push((pacman_pos, path, 0.), 0.)
        closed = set()

        while True:
            if fringe.isEmpty():
                return []

            priority, (current, path, cost) = fringe.pop()

            if current[0] == int(ghost_pos[0]) and current[1] == int(ghost_pos[1]):
                return path

            if current in closed:
                continue

            closed.add(current)

            for inc_y, inc_x, action in self.moves:
                # Check new coordinates validity
                new_y, new_x = current[0] + inc_y, current[1] + inc_x
                if new_x <= 0 or new_y <= 0:
                    continue
                if new_y >= walls.width or new_x >= walls.height:
                    continue

                # Check if new coords is a wall
                if walls[new_y][new_x]: continue
                # Get floyd distance from new position to ghost
                floyd_dist = self.dist[
                    self.cell_to_index[(new_y, new_x)],
                    self.cell_to_index[tuple(ghost_pos)],
                ]

                # Add new path to fringe
                fringe.push(
                    ((new_y, new_x), path + [action], cost + 1),
                    floyd_dist + cost + 1,
                )

        return path

    def bfs(self, pacman_pos, ghost_pos, walls):
        """Given a Pacman's position, a ghost etimated position and the layout,
        returns a list of legal moves to reach the ghost.

        Arguments:
            pacman_pos: The starting position (x, y) of Pacman
            ghost_pos: The target position (x, y) of the ghost
            walls: The W x H grid of the layout's walls

        Returns:
            A list of directions from Pacman’s position to the ghost’s position
            Returns an empty list if no path is found.
        """

        fringe = Queue()
        fringe.push((pacman_pos, []))
        # Closed set to verify that we do not visit the same cell twice
        visited = set()

        # For a none-empty path
        while not fringe.isEmpty():
            current_pos, path = fringe.pop()

            # Verify the current position has not been already visited
            if current_pos in visited:
                continue

            # Add the current position to the visited states
            visited.add(current_pos)

            # Check if pacman reached the ghost estimated position
            if tuple(current_pos) == tuple(ghost_pos):
                # Return the path to the ghost position
                return path

            # Explore neighbors cells
            x, y = current_pos
            for dx, dy, direction in self.moves:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)

                if (0 <= nx < walls.width and 0 <= ny < walls.height and
                        not walls[nx][ny] and next_pos not in visited):
                    fringe.push((next_pos, path + [direction]))

        # No path found
        return []

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        if self.dist is None:
            # Compute the real distance between any pairs of empty cells
            self.dist, self.cell_to_index = floyd_warshall(walls)
            # print("The distance", self.dist)

        # Closest ghost index and its position
        closest_ghost, closest_ghost_pos =\
            self.closes_ghost_positions(beliefs, eaten, position)

        # Update the targetted ghost
        self.update_targetted_ghost(closest_ghost, closest_ghost_pos)

        # TRY A_Star VS BFS
        # Since the layout is quite small, BFS should be not too bad

        # Perform BFS to find the shortest path to the target ghost
        # path_to_ghost = self.bfs(position, closest_ghost_pos, walls)

        # Perform A* to find the shortest path to the target ghost
        path_to_ghost = self.astar(position, closest_ghost_pos, walls)

        if path_to_ghost:
            # Return the first move in the path
            return path_to_ghost[0]
        else:
            # Stop if no path found
            return Directions.STOP

        return Directions.STOP

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
