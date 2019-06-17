import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import deep_environment as environment
from deep_q_learning_agent import DeepQLearningAgent
from deep_q_learning_agent import ExperienceReplayMemory
import utils as utils
from parameters import (N_EXPERIMENTS,
                        MAX_TIME_STEPS_EPISODE,
                        MAX_EPISODES_EXPERIMENT,
                        N_EPISODES_PER_PLOTTING,
                        UPDATE_ITERATIONS,
                        N_BATCH_MEANS,
                        N_RANDOM_EPISODES,
                        EPSILON_MIN,
                        PIECES,
                        )


def solution_time_steps_evolution():
    # Variables Initialization
    env = environment.Env()  # Initializing Environment
    memory = ExperienceReplayMemory()
    agent = DeepQLearningAgent(actions=env.actions, memory=memory)  # Initializing Deep Q-Learning Agent
    display1 = utils.GraphicDisplay(env)  # Initializing Graphic Display
    display4 = utils.GraphicDisplay(env)  # Initializing Graphic Display

    # Data Initialization
    episode_time_steps_list = []
    episode_epsilons = []
    episode_rewards = []
    episode_learning_rate = []

    # Plotting Initialization
    experiment_time_steps = 1

    # Run N episodes
    for episode in range(MAX_EPISODES_EXPERIMENT):
        env.reset_env()  # Running a new Environment
        state = env.initialState
        observation = env.initialObservation
        pieces = env.piecesPicked
        visits = env.visits
        terminated = False
        episode_time_steps = 0
        cumulative_reward = 0
        while not terminated:

            action = agent.get_action(state, observation, pieces)  # Getting next action
            next_state, next_observation, next_pieces, reward, terminated = env.step(state, action)
            agent.memory.add_experience((state, action, observation, pieces, reward, next_state, next_observation, next_pieces, terminated))
            agent.update_q_function_experience_replay()  # Update Q-function from agent

            cumulative_reward += reward

            if experiment_time_steps % UPDATE_ITERATIONS == 0:
                agent.update_q_network()

            if episode_time_steps == MAX_TIME_STEPS_EPISODE:
                terminated = True
            state = next_state
            observation = next_observation
            pieces = next_pieces
            visits = env.visits
            episode_time_steps += 1
            experiment_time_steps += 1

            if episode_time_steps % 100 == 0:
                print(episode_time_steps)

        if episode % N_EPISODES_PER_PLOTTING == 0:  # Plot solution each N_EPISODES_PER_PLOTTING episodes
            print(episode_time_steps)
            plt.figure(1, figsize=(5, 2))
            utils.plot_episode_solution(episode, episode_time_steps)
            plt.figure(2, figsize=(5, 2))
            utils.plot_episode_solution(episode, cumulative_reward)

        # Update Target Q-Network
        agent.update_q_network()

        # Decay exploration and learning rate
        if episode > N_RANDOM_EPISODES and agent.epsilon > EPSILON_MIN and agent.memory.max_size == len(memory):
            agent.decay_exploration()
        agent.decay_learning_rate()

        # Save Data
        episode_time_steps_list.append(episode_time_steps)
        episode_epsilons.append(agent.epsilon)
        episode_rewards.append(cumulative_reward)
        np.save('mlp_weights', agent.targetModel.get_weights())




        # Display Q-Table
        q_table = agent.generate_q_table([0] * len(PIECES))
        display1.step(1, None, q_table)
        q_table = agent.generate_q_table([1] * len(PIECES))
        display4.step(1, None, q_table)
        #if utils.q_values_converged(q_table):
            #break

    # Showing results
    q_table = agent.generate_q_table()
    plt.figure(2)
    plt.close()
    plt.figure(2)
    l_means = utils.data_mean(episode_rewards, N_BATCH_MEANS)
    utils.plot_line_graphic(l_means, 'Episode cumulative reward', 'Reward')
    plt.figure(3)
    utils.plot_line_graphic(episode_epsilons, 'Episode epsilon', 'Exploration')
    plt.figure(4)
    utils.plot_line_graphic(episode_learning_rate, 'Episode learning rate', 'Learning Rate')

    # Saving Variables
    np.save('q_table', q_table)
    np.save('episode_time_steps', episode_time_steps_list)
    np.save('episode_rewards', episode_rewards)
    np.save('episode_rewards_means', episode_rewards)
    np.save('episode_epsilons', episode_epsilons)

    # Waiting time
    time.sleep(1000)


def converged_solution_time_steps_evolution():

    # Run N experiments
    for experiment in range(N_EXPERIMENTS):
        # Variables Initialization
        env = environment.Env()  # Initializing Environment
        memory = ExperienceReplayMemory()
        agent = DeepQLearningAgent(actions=env.actions, memory=memory)  # Initializing Deep Q-Learning Agent

        # Data Initialization
        episode_time_steps_list = []
        episode_epsilons = []
        episode_rewards = []
        episode_learning_rate = []

        # Plotting Initialization
        experiment_time_steps = 1

        # Run N episodes
        for episode in range(MAX_EPISODES_EXPERIMENT):
            env.reset_env()  # Running a new Environment
            state = env.initialState
            observation = env.initialObservation
            pieces = env.piecesPicked
            visits = env.visits
            terminated = False
            episode_time_steps = 0
            cumulative_reward = 0
            while not terminated:

                action = agent.get_action(state, observation, pieces)  # Getting next action
                next_state, next_observation, next_pieces, reward, terminated = env.step(state, action)
                agent.memory.add_experience(
                    (state, action, observation, pieces, reward, next_state, next_observation, next_pieces, terminated))
                agent.update_q_function_experience_replay()  # Update Q-function from agent

                cumulative_reward += reward

                if experiment_time_steps % UPDATE_ITERATIONS == 0:
                    agent.update_q_network()

                if episode_time_steps == MAX_TIME_STEPS_EPISODE:
                    terminated = True
                state = next_state
                observation = next_observation
                pieces = next_pieces
                visits = env.visits
                episode_time_steps += 1
                experiment_time_steps += 1

                if episode_time_steps % 100 == 0:
                    print(episode_time_steps)

            if episode % N_EPISODES_PER_PLOTTING == 0:  # Plot solution each N_EPISODES_PER_PLOTTING episodes
                print(episode_time_steps)
                plt.figure(1, figsize=(5, 2))
                utils.plot_episode_solution(episode, episode_time_steps)
                plt.figure(2, figsize=(5, 2))
                utils.plot_episode_solution(episode, cumulative_reward)

            # Update Target Q-Network
            agent.update_q_network()

            # Decay exploration and learning rate
            if episode > N_RANDOM_EPISODES and agent.epsilon > EPSILON_MIN and agent.memory.max_size == len(memory):
                agent.decay_exploration()
            agent.decay_learning_rate()

            # Save Data
            episode_time_steps_list.append(episode_time_steps)
            episode_epsilons.append(agent.epsilon)
            episode_rewards.append(cumulative_reward)



        # Saving Variables
        l_means = utils.data_mean(episode_rewards, N_BATCH_MEANS)
        np.save('./Results/npy/episode_time_steps', episode_time_steps_list)
        np.save('./Results/npy/episode_rewards', episode_rewards)
        np.save('./Results/npy/episode_rewards_mean', l_means)
        np.save('./Results/npy/episode_epsilons', episode_epsilons)

        pd.DataFrame(episode_time_steps_list).to_csv('./Results/csv/episode_time_steps_%d.csv' % experiment,
                                                     index_label="epis",sep=';')
        pd.DataFrame(episode_rewards).to_csv('./Results/csv/episode_rewards_%d.csv' % experiment,
                                                     index_label="epis", sep=';')
        pd.DataFrame(l_means).to_csv('./Results/csv/episode_rewards_mean_%d.csv' % experiment,
                                                     index_label="epis", sep=';')
        pd.DataFrame(episode_epsilons).to_csv('./Results/csv/episode_epsilons.csv', index_label="epis", sep=';')

        # Clearing plots
        plt.figure(1, figsize=(5, 2))
        plt.clf()
        plt.figure(2, figsize=(5, 2))
        plt.clf()


    # Waiting time
    time.sleep(1000)


if __name__ == "__main__":
    converged_solution_time_steps_evolution()
