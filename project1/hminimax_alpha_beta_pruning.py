import numpy as np
import psutil
import os

from pacman_module.game import Agent, Directions
from pacman_module.util import manhattanDistance
from functools import lru_cache


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
        self.keyState = []
        self.keyDepth = []
        self.maxDepth = 0

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        # Consider Pacman as MAX player
        cache_info = self.hminimax.cache_info()

        # Memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        print("Cache Stats:")
        print(f"Hits: {cache_info.hits}")
        print(f"Misses: {cache_info.misses}")
        print(f"Current Cache Size: {cache_info.currsize}")

        print("\nMemory Usage:")
        print(f"RSS (Resident Set Size): {memory_info.rss / 1024**2:.2f} MB")
        print(f"VMS (Virtual Memory Size): {memory_info.vms / 1024**2:.2f} MB")

        self.maxDepth = 2 * state.getNumFood()
        if self.maxDepth > 5:
            self.maxDepth = 5
        elif self.maxDepth < 2:
            self.maxDepth = 2

        return self.hminimax(state, 1, self.maxDepth, -np.inf, +np.inf, state.getNumFood())[1]

    def cutOff(self, state, maxDepth, nbfoodinit, distPacGhost):
        """Evaluate if one of the cutoff conditions is met to stop recursion.

        Arguments:
            state:          a game state. See API or class `pacman.GameState`.
            maxDepth:       integer, maximum reachable depth of the search tree
            nbfoodinit:     integer, number of food dots at the beginning of the game
            distPacGhost:   integer, distance between Pacman and the ghost

        Returns:
            A boolean value.
        """
        if state.isWin() or state.isLose() or maxDepth < 0 or nbfoodinit > state.getNumFood():
            return True
        elif distPacGhost > 4 and (self.maxDepth - maxDepth) > 3:
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

        min_dist = np.inf

        for i in foodPos:
            dist = manhattanDistance(pacmanPos, i)
            if dist < min_dist:
                min_dist = dist

        return min_dist
    
    def getMinFoodDist(self, state):
        """Given a Pacman game state, returns the minimum distance between 
        two food dots.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A int.
        """
        foodPos = state.getFood().asList()
        distFoodMin = np.inf

        for i in range(len(foodPos)):
            for j in range(len(foodPos)):
                if i != j:
                    distfood = manhattanDistance(foodPos[i], foodPos[j])

                    if distfood < distFoodMin:
                        distFoodMin = distfood
        return distFoodMin              


    def eval(self, state, maxDepth, nbfoodinit):
        """Given a Pacman game state, returns an estimate of the 
            expected utility of the game state.

            Arguments:
            state:          a game state. See API or class `pacman.GameState`.
            maxDepth:       integer, maximum reachable depth of the search tree
            nbfoodinit:     integer, number of food dots at the beginning of the game

            Returns:
            A int.
                
        """

        if state.isWin():
            return 5000 + state.getScore()

        if state.isLose():
            return -5000 + state.getScore()
        
        if nbfoodinit > state.getNumFood():
            # print("FOOOOD INIT")
            if self.maxDepth > (self.maxDepth - maxDepth):
                self.depth_max = (self.maxDepth - maxDepth)
            return (state.getScore() - (self.maxDepth - maxDepth))
        
        distFoodMin = self.getMinFoodDist(state)
        
        distPacFoodMin = self.getClosestFoodDist(state)

        dist = self.getPacmanGhostdist(state)
        
        if state.getNumFood() <= 1:
            return state.getScore() - distPacFoodMin - 3*10*state.getNumFood()
        
        elif dist > 5:
            return state.getScore() - 3.5*distFoodMin - distPacFoodMin -10*state.getNumFood()
        
        else:
            return state.getScore() - 3.5*distFoodMin - 2*distPacFoodMin - 10*state.getNumFood()
    
        

    @lru_cache(maxsize=4096)
    def hminimax(self, state, maxPlayer: bool, maxDepth: int, alpha: float, beta: float, nbfoodinit: int):
        """Given a Pacman game state, returns the best possible move
            using hminimax with alpha-beta pruning.

            Arguments:
            state:      a game state. See API or class `pacman.GameState`.
            maxPlayer:  boolean, 1 means its max's move, 0 means its min's move
            maxDepth:   integer, maximum reachable depth of the search tree
            alpha:      float, best minimum utility score in the current search tree
            beta:       float, best maximum utility score in the current search tree
            nbfoodinit: integer, number of food dots at the beginning of the game

            Returns:
            tuple of 4 elements
                - utility score of the current state
                - legal move as defined in `game.Directions`        
        """

        move = Directions.STOP

        distPacGhost = self.getPacmanGhostdist(state)

        if self.cutOff(state, maxDepth, nbfoodinit, distPacGhost):
            return self.eval(state, maxDepth, nbfoodinit), move
        
        currentKey = key(state,maxPlayer)

        if currentKey in self.keyState:
            index = self.keyState.index(currentKey)
            in_depth = self.keyDepth[index]

            if in_depth > (self.maxDepth - maxDepth):
                self.keyDepth[index] = (self.maxDepth - maxDepth)
            else:
                if maxPlayer == True:
                    return np.inf, Directions.STOP
                else:
                    return -np.inf, Directions.STOP
        else:
            self.keyState.append(currentKey)
            self.keyDepth.append(self.maxDepth - maxDepth)

        # Case of the MAX player (Pacman)
        if maxPlayer:

            rv_max = -np.inf
            for successor, action in state.generatePacmanSuccessors():
                val = self.hminimax(successor, 0, maxDepth - 1, alpha, beta, nbfoodinit)[0]

                if val >= beta:
                    return val, action
                alpha = max(alpha, val)

                if val > rv_max:
                    rv_max = val
                    move = action

            return rv_max, move
        
        # Case of the MIN player (Ghost)
        else:
            rv_min = +np.inf
            for successor, action in state.generateGhostSuccessors(1):
                val = self.hminimax(successor, 1, maxDepth - 1, alpha, beta, nbfoodinit)[0]

                if val <= alpha:
                    return val, action
                beta = min(beta, val)

                if val < rv_min:
                    rv_min = val
                    move = action

            return rv_min, move

