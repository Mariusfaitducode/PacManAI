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


# class PathPoint:
#     def __init__(self, state, pos, weight, parent, action):
#         self.state = state
#         self.pos = pos
#         self.weight = weight
#         self.parent = parent
#         self.action = action


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

    # def chooseNextPosition(self, target, point_to_explore):
    #
    #     next_point = ()
    #
    #     min_score = 10000
    #
    #     for point in point_to_explore:
    #
    #
    #         # Heuristique = chemin parcouru + nombre de points restant
    #
    #         # distance = np.linalg.norm(target - point.pos)
    #         distance = (target[0] - point.pos[0]) ** 2 + (target[1] - point.pos[1]) ** 2
    #
    #         score = point.weight + distance
    #
    #         if score < min_score:
    #             min_score = score
    #             next_point = point
    #
    #     point_to_explore.remove(next_point)
    #     return next_point

    def astar(self, state):

        path = []
        fringe = PriorityQueue()

        score = 0 + state.getNumFood()

        fringe.push((state, path), score)
        closed = set()

        # cost = length(path)
        # heuristic = state.getNumFood()

        while True:

            if fringe.isEmpty():
                print("Fringe empty", path)
                return path

            priority, (current, path) = fringe.pop()

            if current.isWin():
                print(path)
                return path

            current_key = key(current)

            if current_key in closed:
                continue
            else:
                closed.add(current_key)

            for successor, action in current.generatePacmanSuccessors():

                successor_score = len(path) + successor.getNumFood()

                fringe.push((successor, path + [action]), successor_score)

        return path

        # current_pos, _ = key(state)
        #
        # print("Capsule pos :", target)
        # print("Pacman pos :", current_pos)
        #
        # actual_point = PathPoint(state, current_pos, 0, None, None)
        #
        # points_to_explore = []
        # points_already_explored = []
        #
        # i = 0
        #
        # while actual_point.pos != target:
        #
        #     i+=1
        #     print(i)
        #
        #     # if fringe.isEmpty():
        #     #     print("Fringe empty", path)
        #     #     return path
        #
        #     # Get successors
        #
        #     for successor, action in actual_point.state.generatePacmanSuccessors():
        #         # fringe.push((successor, path + [action]))
        #
        #         point_pos, _ = key(successor)
        #
        #         if point_pos not in points_already_explored:
        #
        #             weight = actual_point.weight+1
        #
        #             points_to_explore.append(PathPoint(successor, point_pos, weight, actual_point, action))
        #
        #     # Choose next position
        #
        #     actual_point = self.chooseNextPosition(target, points_to_explore)
        #
        #     points_already_explored.append(actual_point.pos)
        #
        # # Go back
        #
        # final_path = []
        # last_point = actual_point
        #
        # while last_point.parent is not None:
        #
        #     i -= 1
        #     print(i)
        #     print(last_point.pos)
        #
        #     final_path.append(last_point.action)
        #     last_point = last_point.parent
        #
        # print("FINAL PATH")
        # print(final_path)
        #
        # # reversed_final_path = list(final_path.reverse())
        #
        # reversed_final_path = final_path[::-1]
        #
        # print(reversed_final_path)

        return reversed_final_path
