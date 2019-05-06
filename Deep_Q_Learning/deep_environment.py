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
                        OBSERVATION_DIM)


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

        self.environment = np.zeros([10, 10, 2])

        for wall in self.walls:
            self.environment[wall[0], wall[1]] = [1, 0]
        self.environment[self.goalState[0], self.goalState[1]] = [0, 1]

        self.initialObservation = np.zeros([OBSERVATION_DIM, OBSERVATION_DIM, 2])
        for i in range(-1, 2):
            for j in range(-1, 2):
                if INITIAL_STATE[0] + i == -1 or INITIAL_STATE[0] + i == 10 \
                        or INITIAL_STATE[1] + i == -1 or INITIAL_STATE[1] + i == 10:
                    self.initialObservation[i + 1, j + 1] = [1, 0]
                else:
                    self.initialObservation[i + 1, j + 1] = self.environment[INITIAL_STATE[0] + i, INITIAL_STATE[1] + i]

    def reset_env(self):
        self.initialState = np.array(INITIAL_STATE)
        self.goalState = np.array(GOAL_STATE)
        self.reward[tuple(self.goalState)] = GOAL_STATE_REWARD

    def step(self, state, action):
        terminated = False
        new_state, new_observation = self.move(state, action)

        if (new_state == state).all():
            reward = WALL_STATE_REWARD

        elif (new_state == self.goalState).all():
            reward = self.reward[tuple(new_state)]
            terminated = True

        else:
            reward = self.reward[tuple(new_state)]

        return new_state, new_observation, reward + TIME_STEP_REWARD, terminated

    def move(self, state, action):
        new_state = state + self.coordActions[action]
        new_observation = np.zeros([OBSERVATION_DIM, OBSERVATION_DIM, 2])
        if (sum([(new_state == wall).all() for wall in self.walls]) != 0) \
                or not (0 <= new_state[0] < self.height and 0 <= new_state[1] < self.width):
            new_state = state
        for i in range(-1, 2):
            for j in range(-1, 2):
                if new_state[0] + i == -1 or new_state[0] + i == 10 or new_state[1] + j == -1 or new_state[1] + j == 10:
                    new_observation[i + 1, j + 1] = [1, 0]
                else:
                    new_observation[i + 1, j + 1] = self.environment[new_state[0] + i, new_state[1] + j]
        return new_state, new_observation

