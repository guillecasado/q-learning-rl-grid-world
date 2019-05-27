import numpy as np
from parameters import (GRID_WIDTH,
                        GRID_HEIGHT,
                        ACTIONS,
                        COORD_ACTIONS,
                        GOAL_STATE,
                        GOAL_STATE_REWARD,
                        INITIAL_STATE,
                        WALL_STATE_REWARD,
                        TIME_STEP_REWARD,
                        WALLS,
                        OBSERVATION_DIM,
                        PIECES,
                        PIECE_REWARD)


class Env:
    def __init__(self):
        self.height = GRID_HEIGHT
        self.width = GRID_WIDTH
        self.actions = ACTIONS
        self.coordActions = COORD_ACTIONS
        self.walls = WALLS
        self.pieces = PIECES
        self.piecesPicked = np.zeros([len(PIECES)])
        self.initialState = np.array(INITIAL_STATE)
        self.goalState = np.array(GOAL_STATE)

        self.reward = np.zeros([self.height, self.width])
        self.reward[tuple(self.goalState)] = GOAL_STATE_REWARD
        for wall in self.walls:
            self.reward[tuple(wall)] = WALL_STATE_REWARD
        for piece in self.pieces:
            self.reward[tuple(piece)] = PIECE_REWARD

        self.environment = np.zeros([self.height, self.width, 3])

        for wall in self.walls:
            self.environment[wall[0], wall[1]] = [1, 0, 0]
        for piece in self.pieces:
            self.environment[piece[0], piece[1]] = [0, 1, 0]
        self.environment[self.goalState[0], self.goalState[1]] = [0, 0, 1]

        self.initialObservation = np.zeros([OBSERVATION_DIM, OBSERVATION_DIM, 3])
        for i in range(-1, 2):
            for j in range(-1, 2):
                if INITIAL_STATE[0] + i == -1 or INITIAL_STATE[0] + i == self.height \
                        or INITIAL_STATE[1] + i == -1 or INITIAL_STATE[1] + i == self.width:
                    self.initialObservation[i + 1, j + 1] = [1, 0, 0]
                else:
                    self.initialObservation[i + 1, j + 1] = self.environment[INITIAL_STATE[0] + i, INITIAL_STATE[1] + i]

    def reset_env(self):
        self.initialState = np.array(INITIAL_STATE)
        self.goalState = np.array(GOAL_STATE)
        self.reward[tuple(self.goalState)] = GOAL_STATE_REWARD
        self.piecesPicked = np.zeros([len(PIECES)])
        for piece in self.pieces:
            self.environment[piece[0], piece[1]] = [0, 1, 0]

    def step(self, state, action):
        terminated = False
        next_state, next_observation = self.move(state, action)
        reward = TIME_STEP_REWARD

        if np.array_equal(next_state, state):
            reward += WALL_STATE_REWARD

        elif np.array_equal(next_state, self.goalState):
            reward += GOAL_STATE_REWARD
            terminated = True

        elif next_state in self.pieces:
            for i, piece in enumerate(self.pieces):
                if np.array_equal(next_state, piece) and self.piecesPicked[i] == 0:
                    reward += PIECE_REWARD
                    self.piecesPicked[i] = 1
                    self.environment[next_state[0], next_state[1]] = [0, 0, 0]
        else:
            reward += self.reward[tuple(next_state)]

        return next_state, next_observation, self.piecesPicked, reward, terminated

    def move(self, state, action):
        next_state = state + self.coordActions[action]
        next_observation = np.zeros([OBSERVATION_DIM, OBSERVATION_DIM, 3])
        if (sum([np.array_equal(next_state, wall) for wall in self.walls]) != 0) \
                or not (0 <= next_state[0] < self.height and 0 <= next_state[1] < self.width):
            next_state = state
            
        for i in range(-1, 2):
            for j in range(-1, 2):
                if next_state[0] + i == -1 or next_state[0] + i == self.height or next_state[1] + j == -1 or next_state[1] + j == self.width:
                    next_observation[i + 1, j + 1] = [1, 0, 0]
                
                else:
                    next_observation[i + 1, j + 1] = self.environment[next_state[0] + i, next_state[1] + j]
        
        if next_state in self.pieces:
            next_observation[OBSERVATION_DIM//2, OBSERVATION_DIM//2] = [0, 0, 0]
        
        return next_state, next_observation

