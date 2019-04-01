from Q_Learning.q_learning_agent import QLearningAgent
from Q_Learning import utils, environment


def main():
    env = environment.Env()  # Initializing Environment
    display = utils.GraphicDisplay(env)  # Initializing Graphic Display
    q_learning_agent = QLearningAgent(env.actions)  # Initializing Q-Learning Agent
    # Run 1000 episodes
    for episode in range(1000):
        env.reset_env()  # Running a new Environment
        state = env.initialState
        terminated = False
        display.reset_display()
        while not terminated:
            action = q_learning_agent.get_action(state)  # Getting current state action following e-greedy strategy
            new_state, reward, terminated = env.step(state, action)
            q_learning_agent.update_q_function(state, action, reward, new_state)  # Updating Q-function from agent

            if not (state == new_state).all():
                display.step(action, q_learning_agent.q_values_table)

            state = new_state


if __name__ == "__main__":
    main()
