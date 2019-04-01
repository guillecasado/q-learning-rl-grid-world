import numpy as np

# Epsilon-greedy Policy Evaluation Parameters
EPSILON = 0.95
EPSILON_DECAY_RATE = 0.99

# Deep Q-Learning Algorithm Parameters
DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.001

# Experience Replay Memory Parameters
MEMORY_SIZE = 10000
BATCH_SIZE = 24

N_EPISODES_PER_EXPERIMENT = 500
N_EPISODES_PER_PLOTTING = 1
N_EXPERIMENTS = 20

# Fixed Q-Network Strategy parameters
UPDATE_ITERATIONS = 240

UNIT = 50  # pixels per unit
GRID_HEIGHT = 10
GRID_WIDTH = 10
ACTIONS = np.array([0, 1, 2, 3])  # up, down, left, right
COORD_ACTIONS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])  # up, down, left, right
WALLS = np.array(
    [(2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4)]
)
GOAL_STATE_REWARD = 100
WALL_STATE_REWARD = -1
TIME_STEP_REWARD = -0.1
INITIAL_STATE = [9, 1]
GOAL_STATE = [1, 8]
