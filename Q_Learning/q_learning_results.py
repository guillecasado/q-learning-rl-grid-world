import matplotlib.pyplot as plt
from Q_Learning.q_learning_agent import QLearningAgent
from Q_Learning import utils, environment
import numpy as np

N_EPISODES_PER_EXPERIMENT = 150
N_EPISODES_PER_PLOTTING = 2
N_EXPERIMENTS = 20


def solution_time_steps_evolution(agent_class):
    plt.ion()

    env = environment.Env()  # Initializing Environment
    q_learning_agent = agent_class(env.actions)  # Initializing Q-Learning Agent

    # Run N episodes
    for episode in range(N_EPISODES_PER_EXPERIMENT):
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
        if episode % N_EPISODES_PER_PLOTTING == 0:  # Plot solution each 2 episodes
            utils.plot_episode_solution(episode, time_steps)
    plt.yticks(np.arange(0, 450 + 1, 30))
    plt.show(block=True)


def converged_solution_time_steps_evolution(agent_class):
    plt.ion()

    # Run N experiments
    for experiment in range(N_EXPERIMENTS):
        env = environment.Env()  # Initializing Environment
        q_learning_agent = agent_class(env.actions)  # Initializing Q-Learning Agent

        # Run N episodes
        for episode in range(N_EPISODES_PER_EXPERIMENT):
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
    solution_time_steps_evolution(QLearningAgent)
