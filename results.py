import matplotlib.pyplot as plt
import environment
from q_learning_agent import QLearningAgent
import utils

if __name__ == "__main__":
    env = environment.Env()  # Initializing Environment
    q_learning_agent = QLearningAgent(env.actions)  # Initializing Q-Learning Agent
    plt.ion()

    # Run 20 experiments
    for n in range(1, 21):
        env = environment.Env()  # Initializing Environment
        q_learning_agent = QLearningAgent(env.actions)  # Initializing Q-Learning Agent
        # Run 1000 episodes
        for episode in range(1000):
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
        utils.time_steps_converged_solution_vs_experiment(n, time_steps)
    plt.show(block=True)
