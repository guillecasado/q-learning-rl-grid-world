import deep_environment
from deep_q_learning_agent import DeepQLearningAgent
from deep_q_learning_agent import ExperienceReplayMemory
import utils as utils
import numpy as np
import time
from parameters import (UPDATE_ITERATIONS,
                        N_BATCH_MEANS)

import matplotlib.pyplot as plt


def main():
    env = deep_environment.Env()  # Initializing Environment
    display = utils.GraphicDisplay(env)  # Initializing Graphic Display
    memory = ExperienceReplayMemory()
    agent = DeepQLearningAgent(env.actions, memory)  # Initializing Q-Learning Agent
    experiment_time_steps = 0
    # Run 1000 episodes
    for episode in range(1000):
        env.reset_env()  # Running a new Environment
        state = env.initialState
        terminated = False
        display.reset_display()
        while not terminated:
            action = agent.get_action(state)  # Getting current state action following e-greedy strategy
            new_state, reward, terminated = env.step(state, action)
            agent.memory.add_experience((state, action, reward, new_state, terminated))
            agent.update_q_function_experience_replay()  # Updating Q-function from agent
            if experiment_time_steps % UPDATE_ITERATIONS == 0:
                agent.update_q_network()
            if not (state == new_state).all():
                display.step(action)

            state = new_state
        agent.update_q_network()


def test():
    env = deep_environment.Env()  # Initializing Environment
    memory = ExperienceReplayMemory()
    agent = DeepQLearningAgent(env.actions, memory)  # Initializing Q-Learning Agent
    weights = np.load('mlp_weights.npy', allow_pickle=True)
    agent.targetModel.set_weights(weights)
    display1 = utils.GraphicDisplay(env)  # Initializing Graphic Display
    episode_rewards = np.load('episode_rewards.npy')
    episode_epsilons = np.load('episode_epsilons.npy')

    # Showing results
    plt.figure(2)
    plt.close()
    plt.figure(2)
    l_means = utils.data_mean(episode_rewards, N_BATCH_MEANS)
    utils.plot_line_graphic(l_means, 'Episode cumulative reward', 'Reward')
    plt.figure(3)
    utils.plot_line_graphic(episode_epsilons, 'Episode epsilon', 'Exploration')


    """
    # Display Q-Table
    display1 = utils.GraphicDisplay(env)  # Initializing Graphic Display
    q_table = agent.generate_q_table([0, 0, 0])
    display1.step(1, q_table)
    display2 = utils.GraphicDisplay(env)  # Initializing Graphic Display
    q_table = agent.generate_q_table([0, 0, 1])
    display2.step(1, q_table)
    display3 = utils.GraphicDisplay(env)  # Initializing Graphic Display
    q_table = agent.generate_q_table([1, 0, 1])
    display3.step(1, q_table)
    display4 = utils.GraphicDisplay(env)  # Initializing Graphic Display
    q_table = agent.generate_q_table([1, 1, 1])
    display4.step(1, q_table)
    """

    # Run 1000 episodes
    for episode in range(1000):
        env.reset_env()  # Running a new Environment
        state = env.initialState
        observation = env.initialObservation
        pieces = env.piecesPicked
        terminated = False
        display1.reset_display()
        while not terminated:
            action = agent.get_action_test(state, observation, pieces)  # Getting current state action following e-greedy strategy
            next_state, next_observation, next_pieces, reward, terminated = env.step(state, action)
            if not (state == next_state).all():
                display1.step(action, next_state)

            state = next_state
            observation = next_observation
            pieces = next_pieces

            time.sleep(0.1)


if __name__ == "__main__":
    test()
