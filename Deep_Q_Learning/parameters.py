import numpy as np

# Epsilon-greedy Policy Evaluation Parameters
EPSILON = 1
EPSILON_DECAY_RATE = 0.9982
EPSILON_DECREASE_RATE = 0.0025
EPSILON_MIN = 0.2
N_RANDOM_EPISODES = 100
EPSILON_TEST = 0.2

#UCB Policy Evaluation Parameters:
UCB_EXPLORATION = 400


# Deep Q-Learning Algorithm Parameters
DISCOUNT_FACTOR = 0.95

LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 1
LEARNING_RATE_MIN = 0

# Experience Replay Memory Parameters
MEMORY_SIZE = 10000
MEMORY_BATCH_SIZE = 100
EPOCHS = 1
FIT_BATCH_SIZE = None

MAX_EPISODES_EXPERIMENT = 3000
MAX_TIME_STEPS_EPISODE = 1000
N_EPISODES_PER_PLOTTING = 1
N_EXPERIMENTS = 20
N_BATCH_MEANS = 20

# Fixed Q-Network Strategy parameters
UPDATE_ITERATIONS = 1

UNIT = 50  # pixels per unit
GRID_HEIGHT = 10
GRID_WIDTH = 10
ACTIONS = np.array([0, 1, 2, 3])  # up, down, left, right
COORD_ACTIONS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])  # up, down, left, right
GOAL_STATE_REWARD = 100
WALL_STATE_REWARD = -1
PIECE_REWARD = 100
TIME_STEP_REWARD = -1
INITIAL_STATE = [0, 0]
GOAL_STATE = [9, 9]


WALLS = np.array(
    [(0, 5), (1, 5), (2, 5), (2, 0), (2, 1), (2, 2), (3, 8), (3, 9), (5, 4), (5, 5), (5, 6), (5, 7), (7, 2), (8, 2),
     (9, 2)]
)

PIECES = np.array(
    [(2, 6), (4, 2), (7, 1), (9, 3)]
)

OBSERVATION_DIM = 3


# WALLS = np.array(
#     [(0, 2), (0, 5), (1, 2), (1, 5), (2, 2), (2, 7), (2, 8), (2, 9), (3, 2), (4, 2), (4, 5), (5, 5), (6, 0),
#      (6, 8), (6, 9), (7, 2), (7, 3), (7, 4), (7, 5), (9, 3), (7, 6)]
# )

#WALLS5 = np.array(
#    [(0, 1), (1, 1), (2, 3), (3, 0), (3, 1)]
#)

#PIECES5 = np.array(
#    [(1,2), (2, 4), (4, 1)]
#)

# WALLS7 = np.array(
#     [(0, 4), (1, 4), (2, 0), (2, 1), (2, 2), (4, 4), (4, 5), (4, 6), (4, 2), (5, 2)]
# )

# PIECES7 = np.array(
#     [(1, 5), (3, 1), (6, 2), (3, 6)]
# )

# WALLS10 = np.array(
#     [(0, 5), (1, 5), (2, 5), (2, 0), (2, 1), (2, 2), (3, 8), (3, 9), (5, 4), (5, 5), (5, 6), (5, 7), (7, 2), (8, 2),
#      (9, 2)]
# )

#WALLS10_2 = np.array(
#    [(0, 5), (1, 5), (2, 0), (2, 1), (2, 2), (3, 8), (3, 9), (5, 4), (5, 5), (5, 6), (5, 7), (7, 2), (7, 3)]
#)


#PIECES10 = np.array(
#    [(1, 6), (4, 4), (6, 0), (9, 2)]
#)

#)
