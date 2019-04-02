import numpy as np

UNIT = 50  # pixels per unit
GRID_HEIGHT = 10
GRID_WIDTH = 10
ACTIONS = np.array([0, 1, 2, 3])  # up, down, left, right
COORD_ACTIONS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])  # up, down, left, right
GOAL_STATE_REWARD = 100
WALL_STATE_REWARD = -1
TIME_STEP_REWARD = -0.1
INITIAL_STATE = [9, 1]
GOAL_STATE = [1, 8]

WALLS = np.array(
    [(0, 2), (0, 5), (1, 2), (1, 5), (2, 2), (2, 7), (2, 8), (2, 9), (3, 2), (4, 2), (4, 5), (5, 5), (6, 0),
     (6, 8), (6, 9), (7, 2), (7, 3), (7, 4), (7, 5), (9, 3), (7, 6)]
)


class Env:
    def __init__(self):
        self.height = GRID_HEIGHT
        self.width = GRID_WIDTH
        self.actions = ACTIONS
        self.coordActions = COORD_ACTIONS
        self.walls = WALLS
        self.goalState = np.array(GOAL_STATE)
        self.initialState = np.array(INITIAL_STATE)
        while (sum([(self.initialState == wall).all() for wall in self.walls]) != 0) \
                or (self.initialState == self.goalState).all():
            self.initialState = np.array([np.random.randint(9), np.random.randint(9)])

        self.reward = np.zeros([self.height, self.width])
        self.reward[tuple(self.goalState)] = GOAL_STATE_REWARD
        for wall in self.walls:
            self.reward[tuple(wall)] = WALL_STATE_REWARD

    def reset_env(self):
        self.goalState = np.array(GOAL_STATE)
        self.reward[tuple(self.goalState)] = GOAL_STATE_REWARD
        self.initialState = np.array(INITIAL_STATE)

        while (sum([(self.initialState == wall).all() for wall in self.walls]) != 0) \
                or (self.initialState == self.goalState).all():
            self.initialState = np.array([np.random.randint(9), np.random.randint(9)])

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

