import numpy as np

# Epsilon-greedy Policy Evaluation Parameters
EPSILON = 1
EPSILON_DECAY_RATE = 0.975

# Deep Q-Learning Algorithm Parameters
DISCOUNT_FACTOR = 0.98
LEARNING_RATE = 0.01

# Experience Replay Memory Parameters
MEMORY_SIZE = 10000
BATCH_SIZE = 10

MAX_EPISODES_EXPERIMENT = 500
MAX_TIME_STEPS_EPISODE = 2000
N_EPISODES_PER_PLOTTING = 1
N_EXPERIMENTS = 20
N_BATCH_MEANS = 10

# Fixed Q-Network Strategy parameters
UPDATE_ITERATIONS = 1

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

OBSERVATION_DIM = 3


# WALLS = np.array(
#     [(0, 2), (0, 5), (1, 2), (1, 5), (2, 2), (2, 7), (2, 8), (2, 9), (3, 2), (4, 2), (4, 5), (5, 5), (6, 0),
#      (6, 8), (6, 9), (7, 2), (7, 3), (7, 4), (7, 5), (9, 3), (7, 6)]
# )
