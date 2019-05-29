import numpy as np

# Epsilon-greedy Policy Evaluation Parameters
EPSILON = 1
EPSILON_DECAY_RATE = 0.99
EPSILON_MIN = 0.01
N_RANDOM_EPISODES = 400
EPSILON_TEST = 0.2


# Deep Q-Learning Algorithm Parameters
DISCOUNT_FACTOR = 0.9

LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 1
LEARNING_RATE_MIN = 0.01

# Experience Replay Memory Parameters
MEMORY_SIZE = 10000
BATCH_SIZE = 100

MAX_EPISODES_EXPERIMENT = 1000
MAX_TIME_STEPS_EPISODE = 1000
N_EPISODES_PER_PLOTTING = 1
N_EXPERIMENTS = 20
N_BATCH_MEANS = 20

# Fixed Q-Network Strategy parameters
UPDATE_ITERATIONS = 1

UNIT = 50  # pixels per unit
GRID_HEIGHT = 7
GRID_WIDTH = 7
ACTIONS = np.array([0, 1, 2, 3])  # up, down, left, right
COORD_ACTIONS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])  # up, down, left, right
GOAL_STATE_REWARD = 100
WALL_STATE_REWARD = -10
PIECE_REWARD = 100
TIME_STEP_REWARD = -1
INITIAL_STATE = [0, 0]
GOAL_STATE = [6, 6]
WALLS = np.array(
    []
)
PIECES = np.array(
    [(3, 3), (6, 0), (0, 6)]
)

OBSERVATION_DIM = 3


# WALLS = np.array(
#     [(0, 2), (0, 5), (1, 2), (1, 5), (2, 2), (2, 7), (2, 8), (2, 9), (3, 2), (4, 2), (4, 5), (5, 5), (6, 0),
#      (6, 8), (6, 9), (7, 2), (7, 3), (7, 4), (7, 5), (9, 3), (7, 6)]
# )
