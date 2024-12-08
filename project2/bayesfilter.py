import numpy as np
from scipy.stats import binom
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall as sparse_floyd_warshall

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

"""
Bayes filter with A* and Bayes, floyd distance heuristic
Layout               Ghost     #Ghost   Score    Time [s]        Error
-------------------  -------   ------   --------  -------------  -------
large_filter         fearless  1             678         0.1579    --
large_filter         afraid    1             676         0.1647    --
large_filter         terrified 1             654         0.2789    --
-------------------  -------   ------   --------  -------------  -------
large_filter_walls   fearless  1             676         0.1405    --
large_filter_walls   afraid    1             679         0.1436    --
large_filter_walls   terrified 1             648         0.2769    --
-------------------  -------   ------   --------  -------------  -------
zones                fearless  4            1255         0.3878    --
zones                afraid    4            1261         0.4235    --
zones                terrified 4            1251         0.4350    --
-------------------  -------   ------   --------  -------------  -------
grid                 fearless  4            1243         0.4430     --
grid                 afraid    4            1256         0.3212     --
grid                 terrified 4            1226         0.4431     --
-------------------  -------   ------   --------  -------------  -------
"""

"""
Bayes filter with distance reduction with max belief, floyd distance heuristic
Layout               Ghost     #Ghost   Score    Time [s]        Error
-------------------  -------   ------   --------  -------------  -------
large_filter         fearless  1             676         0.0752    --
large_filter         afraid    1             676         0.0731    --
large_filter         terrified 1             673         0.0972    --
-------------------  -------   ------   --------  -------------  -------
large_filter_walls   fearless  1             676         0.0675    --
large_filter_walls   afraid    1             679         0.0637    --
large_filter_walls   terrified 1             682         0.0461    --
-------------------  -------   ------   --------  -------------  -------
zones                fearless  4            1271         0.1781    --
zones                afraid    4            1264         0.1774    --
zones                terrified 4            1266         0.1883    --
-------------------  -------   ------   --------  -------------  -------
grid                 fearless  4            1272         0.1307     --
grid                 afraid    4            1258         0.1819     --
grid                 terrified 4            1255         0.1706     --
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
    # Transform the walls into a numpy array
    walls_array = np.array(walls.data)

    # Possible moves for a cell (going [West, East, North, South])
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    H, W = walls_array.shape
    # list of cells' coordinates in the grid that are not walls
    empty_cells = [(x, y) for x in range(H) for y in range(W)
                   if not walls_array[x, y]]

    # Mapping from cell to index
    cell_to_index = {cell: idx for idx, cell in enumerate(empty_cells)}
    nbr_empty_cells = len(empty_cells)

    # Sparse adjacency matrix
    H_indices = []
    W_indices = []
    data = []

    # Populate adjacency matrix with edges (distance = 1 for neighbors)
    # Determine the neighbors of each cell, and add an edge between them
    for (x, y) in empty_cells:
        current_idx = cell_to_index[(x, y)]
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if (nx, ny) in cell_to_index:
                neighbor_idx = cell_to_index[(nx, ny)]
                H_indices.append(current_idx)
                W_indices.append(neighbor_idx)
                data.append(1)

    # Create sparse matrix representation of the graph
    adjacency_matrix = csr_matrix((data, (W_indices, H_indices)),
                                  shape=(nbr_empty_cells, nbr_empty_cells))

    # Compute the shortest paths using sparse Floyd-Warshall
    dist = sparse_floyd_warshall(adjacency_matrix,
                                 directed=False, unweighted=True)

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
        # The more afraid the ghost is, the larger the value will be
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

        # Definition of the transition matrix with the provided dimensions
        T_matrix = np.zeros((W, H, W, H))

        # Probability of the ghost to be afraid is 2^fear_level
        ghost_fear_prob = 2**self.ghost_dict.get(self.ghost)

        # Precompute Manhattan distances from Pacman's position
        pacman_distances = np.fromfunction(
            lambda x, y: manhattanDistance((x, y), position),
            (W, H),
            dtype=int
        )

        # Loop over each cell of the layout
        for i in range(W):
            for j in range(H):
                # If the cell is a wall, prob = 0
                if walls[i][j]:
                    continue

                # To store the probability of the ghost to be in
                # the 4 possible cells
                prob_ghost = np.zeros(len(moves))

                # Current distance between pacman and the ghost
                current_dist = pacman_distances[i, j]

                # The probability of the ghost to be in another cell
                # Provided the 4 possible moves ([right, left, up, down])
                for counter, (dx, dy) in enumerate(moves):
                    x, y = i + dx, j + dy

                    # Verify if the index is not out of the range
                    # And that it is NOT a wall
                    if 0 <= x < W and 0 <= y < H and not walls[x][y]:

                        # New distance between pacman and the ghost position
                        # After a potential move
                        new_dist = pacman_distances[x, y]

                        # If the ghost is afraid, the probability of the ghost
                        # to be further is 2^fear_level otherwise 1
                        if current_dist <= new_dist:
                            prob_ghost[counter] = ghost_fear_prob
                        else:
                            prob_ghost[counter] = 1

                # Normalize probabilities
                norm_term = prob_ghost.sum()
                if norm_term > 0:
                    prob_ghost /= norm_term

                # Assign probabilities to the transition matrix
                for counter, (dx, dy) in enumerate(moves):
                    x, y = i + dx, j + dy
                    if 0 <= x < W and 0 <= y < H and not walls[x][y]:
                        T_matrix[x, y, i, j] = prob_ghost[counter]

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

        # Precompute Manhattan distances from Pacman's position
        pacman_distances = np.fromfunction(
            lambda x, y: np.abs(x - position[0]) + np.abs(y - position[1]),
            (W, H),
            dtype=int
        )

        # Compute z for all non-wall cells
        mean_bin = self.n * self.p
        z_values = evidence - pacman_distances + mean_bin

        # Identify valid z values (within [0, n]) and not walls
        walls_array = np.array(walls.data)
        valid_z_mask = (z_values >= 0) & (z_values <= self.n) & (~walls_array)

        # print("valid_z_mask", valid_z_mask)
        # Compute binomial probabilities for valid z values
        valid_probs = binom.pmf(z_values[valid_z_mask], self.n, self.p)

        # Assign probabilities to valid cells
        O_matrix[valid_z_mask] = valid_probs

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
                    if x < 0 or x >= W or y < 0 or y >= H or walls[x][y]:
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
        # Index of the targeted ghost
        self.targeted_ghosts = None
        # The position of the targeted ghost
        self.targeted_ghosts_pos = None
        # Remaining attention time to the actual targeted ghost
        # while another ghost has be detected closer
        self.attention_durations = None
        # Possible moves
        self.moves = [
            (-1, 0, Directions.WEST),
            (1, 0, Directions.EAST),
            (0, -1, Directions.SOUTH),
            (0, 1, Directions.NORTH)
        ]

    def closest_ghost_positions(self, beliefs, eaten, position):
        """Provide the index of the closest ghost for which the maximum
        position belief provide the closes distance to pacman's position.

        Arguments:
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A 3-tuple:
            - int representing the ghost identification,
            - int representing a distance.
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
        return closest_ghost, ghost_pos[closest_ghost], ghost_pos

    def update_targeted_ghost(self, closest_ghost, closest_ghost_pos,
                               ghost_pos, eaten):
        """Given the closest ghost at the current step, updates the ghost
        targeted by pacman if a ghost has been closer to pacman for 4
        following steps than its actual targeted.

        Arguments:
            closest_ghost: The index of the closest ghost to pacman.
            closest_ghost_pos: most probable x and y coordinates of the closest ghost.
            ghost_pos: The position the closest ghost.
            eaten: A list of booleans indicating which ghosts have been eaten.

        Returns:
            None
        """
        if self.targeted_ghosts is not None and eaten[self.targeted_ghosts]:
            self.targeted_ghosts = None
            self.targeted_ghosts_pos = None
            self.attention_durations = None

        # Identify the first closest ghost
        if self.targeted_ghosts is None:
            self.targeted_ghosts = closest_ghost
            self.targeted_ghosts_pos = ghost_pos
            self.attention_durations = 4

        # If the closest ghost is not the targeted one
        if self.targeted_ghosts != closest_ghost:
            # If the attention duration is over, change of targeted_ghosts
            if self.attention_durations == 0:
                self.targeted_ghosts = closest_ghost
                self.targeted_ghosts_pos = closest_ghost_pos
                self.attention_durations = 4
            # If the attention duration is not over,
            # decrease the remaining attention duration for that ghost
            else:
                self.attention_durations -= 1
                self.targeted_ghosts_pos = ghost_pos[self.targeted_ghosts]
        else:
            # If the closest ghost is the targeted one,
            # reset the attention duration
            self.attention_durations = 4
            self.targeted_ghosts_pos = ghost_pos[self.targeted_ghosts]

    def astar(self, pacman_pos, ghost_pos, walls):
        """Given Pacman's position, a ghost estimated position and the layout,
        returns a list of legal moves to reach the ghost computed
        using A* with Floyd distances.

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

            if current[0] == int(ghost_pos[0])\
                    and current[1] == int(ghost_pos[1]):
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
                if walls[new_y][new_x]:
                    continue
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

    def astarBelief(self, pacman_pos, ghost_pos, walls, beliefs):
        """Given Pacman's position, a ghost estimated position and the layout,
        returns a list of legal moves to reach the ghost computed using A*
        with Floyd distances.

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

            if current[0] == int(ghost_pos[0])\
                    and current[1] == int(ghost_pos[1]):
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
                if walls[new_y][new_x]:
                    continue
                # Get floyd distance from new position to ghost
                floyd_dist = self.dist[
                    self.cell_to_index[(new_y, new_x)],
                    self.cell_to_index[tuple(ghost_pos)],
                ]

                # Calculate the probability density at the new cell
                density = self.get_density((current[0], current[1]),
                                           inc_x, inc_y,
                                           beliefs[self.targeted_ghosts])
                # Use density to adjust the heuristic
                adjusted_heuristic = density * 1

                # Add new path to fringe
                fringe.push(
                    ((new_y, new_x), path + [action], cost + 1),
                    floyd_dist + cost + 1 + adjusted_heuristic,
                )

        return path

    def bfs(self, pacman_pos, ghost_pos, walls):
        """Given a Pacman's position, a ghost estimated position and the layout,
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

    def find_optimal_move(self, pacPos, walls, beliefs):
        """
        Determines the best move for Pacman based on real distance
        and the belief about the ghost position.

        Arguments:
            pacPos: Pacman's current position.
            walls: The W x H grid of the layout's walls
            beliefs: The list of current ghost belief states.

        Returns:
            The optimal move direction
        """
        x, y = pacPos
        optimal_move = Directions.STOP

        # Compute moves that minimize the real distance to the ghost
        possible_moves = []
        min_distance = np.inf

        for dx, dy, move in self.moves:
            if walls[x + dx][y + dy]:
                continue

            new_pos = (x + dx, y + dy)
            ghost_pos = self.targeted_ghosts_pos

            distance = self.dist[
                self.cell_to_index[tuple(new_pos)],
                self.cell_to_index[tuple(ghost_pos)]
            ]
            possible_moves.append((distance, dx, dy, move))
            min_distance = min(min_distance, distance)

        # Keep only the moves that minimize the distance to the ghost
        optimal_moves = [(dx, dy, move) for dist, dx, dy, move
                         in possible_moves if dist == min_distance]

        if len(optimal_moves) == 1:
            # If only one optimal move exists, return it
            return optimal_moves[0][2]

        # Choose the move leading to the highest probability
        # of finding the ghost based on the belief
        max_density = - np.inf

        for dx, dy, move in optimal_moves:
            if walls[x + dx][y + dy]:
                continue

            # Compute the density in the direction of the move
            density = self.get_density(pacPos, dx, dy,
                                       beliefs[self.targeted_ghosts])

            # Look for the move with the highest density
            if density > max_density:
                max_density = density
                optimal_move = move

        return optimal_move

    def get_density(self, pac_pos, dx, dy, belief):
        """
       Computes the probability of finding the targeted ghost in a
       given direction.

        Arguments:
            pac_pos: The current position of Pacman (x, y).
            dx: The change in the x-coordinate for the move.
            dy: The change in the y-coordinate for the move.
            belief: The list of current ghost belief states.

        Returns:
            density: The total probability (density) in the
                    direction specified by dx and dy.
        """
        W, H = belief.shape
        x, y = pac_pos

        density = 0

        # If the move is in the x direction (dx is non-zero)
        if dx != 0:
            # defines a range of x-coordinates for iteration
            # based on the movement direction
            x_range = range(x + dx, W if dx > 0 else -1, dx)
            for i in x_range:
                starty = abs(x - i)
                y_min = max(y - starty, 0)
                y_max = min(y + starty + 1, H)
                # Sum the slice of the row i of the belief matrix
                # covering columns from y_min to y_max
                density += belief[i, y_min:y_max].sum()

        # If the move is in the y direction (dy is non-zero)
        elif dy != 0:
            # defines a range of x-coordinates for iteration
            # based on the movement direction
            y_range = range(y + dy, H if dy > 0 else -1, dy)
            for j in y_range:
                startx = abs(y - j)
                x_min = max(x - startx, 0)
                x_max = min(x + startx + 1, W)
                # Sum the slice of the column j of the belief matrix
                # covering columns from x_min to y_max
                density += belief[x_min:x_max, j].sum()

        return density

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
        closest_ghost, closest_ghost_pos, ghost_pos =\
            self.closest_ghost_positions(beliefs, eaten, position)

        # Update the targeted ghost
        self.update_targeted_ghost(closest_ghost,
                                    closest_ghost_pos, ghost_pos, eaten)

        # Perform BFS to find the shortest path to the target ghost
        # path_to_ghost = self.bfs(position, closest_ghost_pos, walls)

        # Perform A* to find the shortest path to the target ghost
        # path_to_ghost = self.astar(position, closest_ghost_pos, walls)

        # Perform A*  using the belief to find the
        # shortest path to the target ghost
        # path_to_ghost = self.astarBelief(position,
        #                       closest_ghost_pos, walls, beliefs)

        # if path_to_ghost:
        #     # Return the first move in the path
        #     return path_to_ghost[0]
        # else:
        #     # Stop if no path found
        #     return Directions.STOP

        # return Directions.STOP

        optimal_move = self.find_optimal_move(position, walls, beliefs)

        return optimal_move

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
