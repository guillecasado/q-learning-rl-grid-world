import numpy as np
import random

EPSILON = 0.9
EPSILON_DECAY_RATE = 0.99
DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.01


class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.epsilon = EPSILON
        self.q_values_table = np.zeros([10, 10, 4])

    def update_q_function(self, state, action, reward, next_state):
        # Current Q-value
        current_q = self.q_values_table[state[0], state[1], action]

        # New Q-value
        new_q = reward + self.discount_factor * max(self.q_values_table[next_state[0], next_state[1]])

        # Updated Q-function
        self.q_values_table[state[0], state[1], action] += self.learning_rate * (new_q - current_q)

    def get_action(self, state):
        self.epsilon *= EPSILON_DECAY_RATE
        if np.random.rand() <= self.epsilon:
            return random.choice(self.actions)
        else:
            return self.arg_max(self.q_values_table[state[0], state[1]])

    @staticmethod
    def arg_max(state_q_values):
        max_actions_list = []
        max_q_value = state_q_values[0]
        for i, q_value in enumerate(state_q_values):
            if q_value > max_q_value:
                max_q_value = q_value
                max_actions_list.clear()
                max_actions_list.append(i)
            elif q_value == max_q_value:
                max_actions_list.append(i)
        return random.choice(max_actions_list)




