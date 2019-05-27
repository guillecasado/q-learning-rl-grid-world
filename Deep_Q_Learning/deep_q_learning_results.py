import time
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
                        EPSILON_MIN)


def solution_time_steps_evolution():
    plt.ion()

    env = environment.Env()  # Initializing Environment
    memory = ExperienceReplayMemory()
    agent = DeepQLearningAgent(actions=env.actions, memory=memory)  # Initializing Deep Q-Learning Agent
    experiment_time_steps = 1

    display = utils.GraphicDisplay(env)  # Initializing Graphic Display

    # Run N episodes
    for episode in range(MAX_EPISODES_EXPERIMENT):
        env.reset_env()  # Running a new Environment
        state = env.initialState
        terminated = False
        episode_time_steps = 1
        while not terminated:
            action = agent.get_action(state)  # Getting current state action following e-greedy strategy
            new_state, reward, terminated = env.step(state, action)
            agent.memory.add_experience((state, action, reward, new_state, terminated))
            agent.update_q_function_experience_replay()  # Update Q-function from agent

            if experiment_time_steps % UPDATE_ITERATIONS == 0:
                agent.update_q_network()

            if episode_time_steps % 100 == 0:
                print(episode_time_steps)

            if episode_time_steps == MAX_TIME_STEPS_EPISODE:
                terminated = True

            state = new_state
            episode_time_steps += 1

        if episode % N_EPISODES_PER_PLOTTING == 0:  # Plot solution each 2 episodes
            print(episode_time_steps-1)
            utils.plot_episode_solution(episode, episode_time_steps-1)
    q_table = agent.generate_q_table()
    display.step(1, q_table=q_table)
    time.sleep(100)
    plt.show(block=True)


def deep_q_table_solution():
    # Variables Initialization
    env = environment.Env()  # Initializing Environment
    memory = ExperienceReplayMemory()
    agent = DeepQLearningAgent(actions=env.actions, memory=memory)  # Initializing Deep Q-Learning Agent
    display = utils.GraphicDisplay(env)  # Initializing Graphic Display

    # Data Initialization
    episode_epsilons = []
    episode_rewards = []

    # Plotting Initialization
    plt.figure(1)
    experiment_time_steps = 1

    # Run N episodes
    for episode in range(MAX_EPISODES_EXPERIMENT):
        env.reset_env()  # Running a new Environment
        state = env.initialState
        observation = env.initialObservation
        pieces = env.piecesPicked
        terminated = False
        episode_time_steps = 1
        cumulative_reward = 0
        while not terminated:

            action = agent.get_action(state, observation, pieces)  # Getting current state action following e-greedy strategy
            next_state, next_observation, next_pieces, reward, terminated = env.step(state, action)
            agent.memory.add_experience((state, action, observation, pieces, reward, next_state, next_observation, next_pieces, terminated))
            agent.update_q_function_experience_replay()  # Update Q-function from agent

            cumulative_reward += reward

            # if experiment_time_steps % UPDATE_ITERATIONS == 0:
                # agent.update_q_network()

            if episode_time_steps % 100 == 0:
                print(episode_time_steps)

            if episode_time_steps == MAX_TIME_STEPS_EPISODE:
                terminated = True
            state = next_state
            observation = next_observation
            pieces = next_pieces
            episode_time_steps += 1
            experiment_time_steps += 1

        if episode % N_EPISODES_PER_PLOTTING == 0:  # Plot solution each N_EPISODES_PER_PLOTTING episodes
            print(episode_time_steps)
            plt.figure(1)
            utils.plot_episode_solution(episode, episode_time_steps)
            plt.figure(2)
            utils.plot_episode_solution(episode, cumulative_reward)

        # Decay exploration and update Q-target network
        if episode > N_RANDOM_EPISODES and agent.epsilon > EPSILON_MIN:
            agent.decay_exploration()

        # Save Data
        episode_epsilons.append(agent.epsilon)
        episode_rewards.append(cumulative_reward)

        # Update Target Q-Network
        agent.update_q_network()

        # Display Q-Table
        q_table = agent.generate_q_table()
        display.step(1, q_table)
        np.save('q_table', q_table)
        #if utils.q_values_converged(q_table):
            #break

    # Showing results
    q_table = agent.generate_q_table()
    display.step(1, q_table)
    plt.figure(2)
    plt.close()
    plt.figure(2)
    l_means = utils.data_mean(episode_rewards, N_BATCH_MEANS)
    utils.plot_line_graphic(l_means, 'Episode cumulative reward', 'Reward')
    plt.figure(3)
    utils.plot_line_graphic(episode_epsilons, 'Episode epsilon', 'Exploration')

    input()

    # Saving Variables
    np.save('q_table', q_table)
    np.save('episode_rewards', episode_rewards)
    np.save('episode_epsilons', episode_epsilons)

    # Waiting time
    time.sleep(1000)


def converged_solution_time_steps_evolution(agent_class):
    plt.ion()

    # Run N experiments
    for experiment in range(N_EXPERIMENTS):
        env = environment.Env()  # Initializing Environment
        q_learning_agent = agent_class(env.actions)  # Initializing Q-Learning Agent
        time_steps = 0

        # Run N episodes
        for episode in range(MAX_EPISODES_EXPERIMENT):
            env.reset_env()  # Running a new Environment
            state = env.initialState
            terminated = False
            time_steps = 0
            while not terminated:
                action = q_learning_agent.get_action(state)  # Getting current state action following e-greedy strategy
                new_state, reward, terminated = env.step(state, action)
                q_learning_agent.update_q_function(state, action, reward, new_state)  # Updating Q-function from agent
                state = new_state
                time_steps += 1
        utils.plot_converged_solution(experiment, time_steps)
    plt.show(block=True)


if __name__ == "__main__":
    deep_q_table_solution()
