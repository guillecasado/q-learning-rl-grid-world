import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from Deep_Q_Learning.utils import normalize
from Deep_Q_Learning.parameters import (EPSILON,
                                        EPSILON_DECAY_RATE,
                                        DISCOUNT_FACTOR,
                                        LEARNING_RATE,
                                        MEMORY_SIZE,
                                        BATCH_SIZE)


class DeepQLearningAgent:
    def __init__(self, actions, memory):
        self.actions = actions
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.epsilon = EPSILON

        self.policyModel = self._build_model()
        self.targetModel = self._build_model()

        self.memory = memory

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24,
                                     input_shape=(2,),
                                     activation=keras.activations.relu))
        model.add(keras.layers.Dense(24,
                                     activation=keras.activations.relu))
        model.add(keras.layers.Dense(len(self.actions),
                                     activation=keras.activations.linear))
        model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE), loss='mse')

        return model

    def update_q_function_experience_replay(self):
        if len(self.memory) < self.memory.batch_size:
            return

        # Train the Q-Network by experience replay
        for state, action, reward, next_state, terminated in self.memory.sample():
            state = np.reshape(normalize(state), [1, 2])
            next_state = np.reshape(normalize(next_state), [1, 2])
            # Current Q-values
            q_values = self.policyModel.predict(state)

            # Q-values next step update
            q_update = reward
            if not terminated:
                q_values_next_state = self.targetModel.predict(next_state)[0]
                max_action_next_state = self.arg_max(q_values_next_state)
                q_update = reward + self.discount_factor * q_values_next_state[max_action_next_state]

            # New Q-values
            q_values[0][action] = q_update

            # Updated policy Q-Network
            self.policyModel.fit(state, q_values, verbose=0)

        # Decay the exploration rate
        self.epsilon *= EPSILON_DECAY_RATE

    def update_q_network(self):
        self.targetModel.set_weights(self.policyModel.get_weights())

    def get_action(self, state):
        state = np.reshape(normalize(state), [1, 2])
        if np.random.rand() <= self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.policyModel.predict(state)
            return self.arg_max(q_values[0])

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

    def generate_q_table(self):
        q_values_table = np.zeros([10, 10, 4])
        for i in range(10):
            for j in range(10):
                for a in range(4):
                    q_values_table[i, j, a] = self.targetModel.predict(np.reshape(normalize([i,j]), [1, 2]))[0][a]
        return q_values_table


class ExperienceReplayMemory:
    def __init__(self):
        self.max_size = MEMORY_SIZE
        self.memory = []
        self.batch_size = BATCH_SIZE

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def add_experience(self, experience_tuple):
        self.memory.append(experience_tuple)
        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def __len__(self):
        return len(self.memory)
