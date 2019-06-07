import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from utils import normalize
from parameters import (EPSILON,
                        EPSILON_DECAY_RATE,
                        EPSILON_TEST,
                        DISCOUNT_FACTOR,
                        LEARNING_RATE,
                        MEMORY_SIZE,
                        MEMORY_BATCH_SIZE,
                        OBSERVATION_DIM,
                        PIECES,
                        GRID_HEIGHT,
                        GRID_WIDTH,
                        LEARNING_RATE_DECAY,
                        UCB_EXPLORATION,
                        EPOCHS,
                        FIT_BATCH_SIZE)

import deep_environment as environment

class ExperienceReplayMemory:
    def __init__(self):
        self.max_size = MEMORY_SIZE
        self.memory = []
        self.batch_size = MEMORY_BATCH_SIZE

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def add_experience(self, experience_tuple):
        self.memory.append(experience_tuple)
        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def __len__(self):
        return len(self.memory)

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
        model.add(keras.layers.Dense(50,
                                     input_shape=(
                                         OBSERVATION_DIM*OBSERVATION_DIM * 3 +
                                         len(PIECES) +
                                         2,),
                                     activation=keras.activations.sigmoid))
        model.add(keras.layers.Dense(len(self.actions),
                                     activation=keras.activations.linear))
        model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE), loss='mse')

        return model

    def update_q_function_experience_replay(self):

        if len(self.memory) < self.memory.batch_size:
            return

        mlp_states_array = np.zeros([self.memory.batch_size,
                                     OBSERVATION_DIM * OBSERVATION_DIM * 3 +
                                     len(PIECES) +
                                     2])
        mlp_next_states_array = np.zeros([self.memory.batch_size,
                                          OBSERVATION_DIM * OBSERVATION_DIM * 3 +
                                          len(PIECES) +
                                          2])

        inputs = np.zeros([self.memory.batch_size,
                           OBSERVATION_DIM * OBSERVATION_DIM * 3 +
                           len(PIECES) +
                           2])
        targets = np.zeros([self.memory.batch_size, len(self.actions)])

        mem_samples = self.memory.sample()

        # Train the Q-Network by experience replay
        for i, sample in enumerate(mem_samples):
            state, action, obs, pieces, reward, next_state, next_obs, next_pieces, terminated = sample
            state = np.reshape(normalize(state), [1, 2])
            pieces = np.reshape(pieces, [1, len(PIECES)])
            observation = np.reshape(obs, [1, OBSERVATION_DIM * OBSERVATION_DIM * 3])
            next_state = np.reshape(normalize(next_state), [1, 2])
            next_observation = np.reshape(next_obs, [1, OBSERVATION_DIM * OBSERVATION_DIM * 3])
            next_pieces = np.reshape(next_pieces, [1, len(PIECES)])
            mlp_state = np.reshape(np.concatenate([
                pieces[0],
                observation[0],
                state[0]]),
                [1,
                 OBSERVATION_DIM * OBSERVATION_DIM * 3 +
                 len(PIECES) +
                 len(state[0])])
            mlp_states_array[i] = mlp_state

            mlp_next_state = np.reshape(np.concatenate([
                next_pieces[0],
                next_observation[0],
                next_state[0]]),
                [1,
                 OBSERVATION_DIM * OBSERVATION_DIM * 3 +
                 len(PIECES) +
                 len(state[0])])
            mlp_next_states_array[i] = mlp_next_state

            inputs[i] = mlp_state

        state_preds = self.policyModel.predict(mlp_states_array)
        next_state_preds = self.policyModel.predict(mlp_next_states_array)

        for i, (q_values, next_q_values, sample) in enumerate(zip(state_preds, next_state_preds, mem_samples)):

            state, action, obs, pieces, reward, next_state, next_obs, next_pieces, terminated = sample

            targets[i] = q_values

            q_update = reward
            if not terminated:
                max_action_next_state = self.arg_max(next_q_values)
                q_update = reward + self.discount_factor * next_q_values[max_action_next_state]

            # New Q-values
            targets[i][action] = q_update
        # Updated policy Q-Network
        self.policyModel.fit(inputs, targets, epochs=EPOCHS, batch_size=FIT_BATCH_SIZE,verbose=0)

    def update_q_network(self):
        self.targetModel.set_weights(self.policyModel.get_weights())

    def decay_exploration(self):
        self.epsilon *= EPSILON_DECAY_RATE

    def decay_learning_rate(self):
        self.learning_rate *= LEARNING_RATE_DECAY

    def get_action(self, state, observation, pieces):
        state = np.reshape(normalize(state), [1, 2])
        observation = np.reshape(observation, [1, OBSERVATION_DIM * OBSERVATION_DIM * 3])
        pieces = np.reshape(pieces, [1, len(PIECES)])

        mlp_state = np.reshape(np.concatenate([
            pieces[0],
            observation[0],
            state[0]]),
            [1,
             OBSERVATION_DIM * OBSERVATION_DIM * 3 +
             len(PIECES) +
             2])
        if np.random.rand() <= self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.policyModel.predict(mlp_state)
            return self.arg_max(q_values[0])

    def get_action_ucb(self, state, observation, pieces, visits):

        ucb_values = np.zeros([len(self.actions)])
        for i, visit in enumerate(visits[state[0], state[1]]):
            ucb_values[i] = UCB_EXPLORATION * np.sqrt(2*(np.log(np.sum(visits[state[0], state[1]]))) / visit)

        state = np.reshape(normalize(state), [1, 2])
        observation = np.reshape(observation, [1, OBSERVATION_DIM * OBSERVATION_DIM * 3])
        pieces = np.reshape(pieces, [1, len(PIECES)])

        mlp_state = np.reshape(np.concatenate([
            pieces[0],
            observation[0],
            state[0]]),
            [1,
             OBSERVATION_DIM * OBSERVATION_DIM * 3 +
             len(PIECES) +
             2])

        q_values = self.policyModel.predict(mlp_state) + ucb_values
        return self.arg_max(q_values[0])

    def get_action_test(self, state, observation, pieces):

        state = np.reshape(normalize(state), [1, 2])
        observation = np.reshape(observation, [1, OBSERVATION_DIM * OBSERVATION_DIM * 3])
        pieces = np.reshape(pieces, [1, len(PIECES)])

        mlp_state = np.reshape(np.concatenate([
            pieces[0],
            observation[0],
            state[0]]),
            [1,
             OBSERVATION_DIM * OBSERVATION_DIM * 3 +
             len(PIECES) +
             2])
        if np.random.rand() <= EPSILON_TEST:
            return random.choice(self.actions)
        else:
            q_values = self.targetModel.predict(mlp_state)
            return self.arg_max(q_values[0])

    def get_action_possible(self, state, observation, pieces):
        possible_actions = []
        i = OBSERVATION_DIM//2
        j = OBSERVATION_DIM-1
        if observation[0, i, 0] == 0:
            possible_actions.append(0)
        if observation[j, i, 0] == 0:
            possible_actions.append(1)
        if observation[i, 0, 0] == 0:
            possible_actions.append(2)
        if observation[i, j, 0] == 0:
            possible_actions.append(3)

        state = np.reshape(normalize(state), [1, 2])
        observation = np.reshape(observation, [1, OBSERVATION_DIM * OBSERVATION_DIM * 3])
        pieces = np.reshape(pieces, [1, len(PIECES)])

        mlp_state = np.reshape(np.concatenate([
            pieces[0],
            observation[0],
            state[0]]),
            [1,
             OBSERVATION_DIM * OBSERVATION_DIM * 3 +
             len(PIECES) +
             2])
        if np.random.rand() <= self.epsilon:
            return random.choice(possible_actions)
        else:
            q_values = self.policyModel.predict(mlp_state)
            return self.arg_max_possible(q_values[0], possible_actions)

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

    @staticmethod
    def arg_max_possible(state_q_values, possible_actions):
        max_actions_list = []
        max_q_value = state_q_values[possible_actions[0]]
        for i, q_value in enumerate(state_q_values):
            if q_value > max_q_value and i in possible_actions:
                max_q_value = q_value
                max_actions_list.clear()
                max_actions_list.append(i)
            elif q_value == max_q_value and i in possible_actions:
                max_actions_list.append(i)
        return random.choice(max_actions_list)

    def generate_q_table(self, pieces_picked=None):
        q_values_table = np.zeros([GRID_HEIGHT, GRID_WIDTH, 4])
        inputs = np.zeros([GRID_HEIGHT*GRID_WIDTH,
                           OBSERVATION_DIM * OBSERVATION_DIM * 3 +
                           len(PIECES) +
                           2])

        env = environment.Env()
        pieces = env.piecesPicked
        if not(pieces_picked is None):
            pieces = pieces_picked
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                observation = np.zeros([OBSERVATION_DIM, OBSERVATION_DIM, 3])
                for i2 in range(-1, 2):
                    for j2 in range(-1, 2):
                        if i + i2 == -1 or i + i2 == GRID_HEIGHT or j + j2 == -1 or j + j2 == GRID_WIDTH:
                            observation[i2 + 1, j2 + 1] = [1, 0, 0]
                        else:
                            observation[i2 + 1, j2 + 1] = env.environment[i + i2, j + j2]
                state = np.reshape(normalize([i, j]), [1, 2])
                observation = np.reshape(observation, [1, OBSERVATION_DIM * OBSERVATION_DIM * 3])
                pieces = np.reshape(pieces, [1, len(PIECES)])
                mlp_state = np.reshape(np.concatenate([
                    pieces[0],
                    observation[0],
                    state[0]]),
                    [1,
                     OBSERVATION_DIM * OBSERVATION_DIM * 3 +
                     len(PIECES) +
                     2])
                inputs[GRID_HEIGHT * i + j] = mlp_state

        predictions = self.targetModel.predict(inputs)
        for m, prediction in enumerate(predictions):
            for a in self.actions:
                q_values_table[m // GRID_HEIGHT, m % GRID_WIDTH, a] = prediction[a]
        return q_values_table

