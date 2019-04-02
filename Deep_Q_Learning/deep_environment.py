import numpy as np
from Deep_Q_Learning.parameters import (GRID_WIDTH,
                                        GRID_HEIGHT,
                                        ACTIONS,
                                        COORD_ACTIONS,
                                        GOAL_STATE,
                                        GOAL_STATE_REWARD,
                                        INITIAL_STATE,
                                        WALL_STATE_REWARD,
                                        TIME_STEP_REWARD,
                                        WALLS)


class Env:
    def __init__(self):
        self.height = GRID_HEIGHT
        self.width = GRID_WIDTH
        self.actions = ACTIONS
        self.coordActions = COORD_ACTIONS
        self.walls = WALLS

        self.initialState = np.array(INITIAL_STATE)
        self.goalState = np.array(GOAL_STATE)

        self.reward = np.zeros([self.height, self.width])
        self.reward[tuple(self.goalState)] = GOAL_STATE_REWARD
        for wall in self.walls:
            self.reward[tuple(wall)] = WALL_STATE_REWARD

    def reset_env(self):
        self.initialState = np.array(INITIAL_STATE)
        self.goalState = np.array(GOAL_STATE)
        self.reward[tuple(self.goalState)] = GOAL_STATE_REWARD

    def step(self, state, action):
        terminated = False
        new_state = self.move(state, action)

        if (sum([(new_state == wall).all() for wall in self.walls]) != 0) \
                or not(0 <= new_state[0] < self.height and 0 <= new_state[1] < self.width):
            reward = WALL_STATE_REWARD
            new_state = state

        elif (new_state == self.goalState).all():
            reward = self.reward[tuple(new_state)]
            terminated = True

        else:
            reward = self.reward[tuple(new_state)]

        return new_state, reward+TIME_STEP_REWARD, terminated

    def move(self, state, action):
        new_state = state + self.coordActions[action]
        return new_state

