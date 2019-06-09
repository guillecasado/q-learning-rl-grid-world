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

        self.visits = np.ones([GRID_HEIGHT, GRID_WIDTH, len(self.actions)])

    def reset_env(self):
        self.initialState = np.array(INITIAL_STATE)
        self.goalState = np.array(GOAL_STATE)
        self.reward[tuple(self.goalState)] = GOAL_STATE_REWARD
        self.piecesPicked = np.zeros([len(PIECES)])
        for piece in self.pieces:
            self.environment[piece[0], piece[1]] = [0, 1, 0]

    def step(self, state, action):
        terminated = False
        next_state, piece_picked = self.move(state, action)

        if np.array_equal(next_state, state):
            reward = WALL_STATE_REWARD

        elif np.array_equal(next_state, self.goalState):
            reward = GOAL_STATE_REWARD*(sum(self.piecesPicked))
            terminated = True

        elif piece_picked:
            for i, piece in enumerate(self.pieces):
                if np.array_equal(next_state, piece) and self.piecesPicked[i] == 0:
                    reward = PIECE_REWARD
                    self.piecesPicked[i] = 1
                    self.environment[next_state[0], next_state[1]] = [0, 0, 0]
        else:
            reward = TIME_STEP_REWARD

        return next_state, self.piecesPicked, reward, terminated

    def move(self, state, action):
        next_state = state + self.coordActions[action]
        piece_picked = False
        if (sum([np.array_equal(next_state, wall) for wall in self.walls]) != 0) \
                or not (0 <= next_state[0] < self.height and 0 <= next_state[1] < self.width):
            next_state = state
        
        for i, piece in enumerate(self.pieces):
            if np.array_equal(piece, next_state) and self.piecesPicked[i] == 0:
                piece_picked = True

        self.visits[state[0], state[1], action] += 1

        return next_state, piece_picked

