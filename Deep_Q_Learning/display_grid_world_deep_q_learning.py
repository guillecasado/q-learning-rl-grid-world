from Deep_Q_Learning import deep_environment
from Deep_Q_Learning.deep_q_learning_agent import DeepQLearningAgent
from Deep_Q_Learning.deep_q_learning_agent import ExperienceReplayMemory
import Deep_Q_Learning.utils as utils
from Deep_Q_Learning.parameters import UPDATE_ITERATIONS


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


if __name__ == "__main__":
    main()
