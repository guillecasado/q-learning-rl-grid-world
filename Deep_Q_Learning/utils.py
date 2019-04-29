import numpy as np
import tkinter as tk
import time
import matplotlib.pyplot as plt
from parameters import (UNIT, COORD_ACTIONS)


# Grid world real-time display class
class GraphicDisplay(tk.Tk):
    def __init__(self, env):
        tk.Tk.__init__(self)
        self.env = env
        self.title('Environment')
        self.geometry('{0}x{1}'.format(self.env.width * UNIT, self.env.height * UNIT))
        self.resizable(False, False)
        self.canvas, self.agent = self.create_canvas()
        self.texts = []
        self.coordActions = COORD_ACTIONS

    def create_canvas(self):
        # Initialize the Canvas
        canvas = tk.Canvas(self, bg='white', highlightthickness=0.1, highlightbackground="black",
                           height=self.env.height * UNIT,
                           width=self.env.width * UNIT)

        # Create Grids
        for col in range(0, self.env.width * UNIT, UNIT):
            x0, y0, x1, y1 = col, 0, col, self.env.height*UNIT
            canvas.create_line(x0, y0, x1, y1)

        for row in range(0, self.env.height * UNIT, UNIT):
            x0, y0, x1, y1 = 0, row, self.env.width*UNIT, row
            canvas.create_line(x0, y0, x1, y1)

        # Create Walls
        for wall in self.env.walls:
            wall_pos = wall * UNIT
            canvas.create_rectangle(wall_pos[1], wall_pos[0],
                                    wall_pos[1]+UNIT, wall_pos[0]+UNIT,
                                    fill='dimgray')

        # Create Initial State
        initial_state = self.env.initialState * UNIT
        agent = canvas.create_rectangle(initial_state[1] + UNIT/3,
                                        initial_state[0] + UNIT/3,
                                        initial_state[1] + UNIT - UNIT/3,
                                        initial_state[0] + UNIT - UNIT/3,
                                        fill='blue')

        # Create Goal State
        goal_state = self.env.goalState * UNIT
        canvas.create_rectangle(goal_state[1], goal_state[0],
                                goal_state[1] + UNIT, goal_state[0] + UNIT, fill='green')
        canvas.pack()
        return canvas, agent

    def text_q_value(self, row, col, value, action, font='Helvetica', size=7, style='normal', anchor="nw"):
        if (sum([((row, col) == wall).all() for wall in self.env.walls])) == 0 \
                and ((row, col) != self.env.goalState).any():
            text_position = np.array([(UNIT*(col+1/2.4), UNIT*(row+1/20)),
                                      (UNIT*(col+1/2.4), UNIT*(row+1/1.25)),
                                      (UNIT*(col+1/20), UNIT*(row+1/2.5)),
                                      (UNIT*(col+1/1.4), UNIT*(row+1/2.55))])
            x, y = text_position[action]
            font = (font, str(size), style)
            text = self.canvas.create_text(x, y, fill="black", text=value, font=font, anchor=anchor)
            return self.texts.append(text)

    def print_q_values(self, q_values_table):

        # Delete old values
        for i in self.texts:
            self.canvas.delete(i)
        self.texts.clear()

        # Print new Q-values
        for i in range(self.env.height):
            for j in range(self.env.width):
                for action in self.env.actions:
                    self.text_q_value(i, j, round(q_values_table[i, j, action], 2), action)

    def step(self, action, q_table=None):
        agent_move = [self.coordActions[action][1]*UNIT, self.coordActions[action][0] * UNIT]
        self.canvas.move(self.agent, agent_move[0], agent_move[1])
        if not(q_table is None):
            self.print_q_values(q_table)
        self.canvas.tag_raise(self.agent)
        time.sleep(0.001)
        self.update()

    def reset_display(self):
        self.canvas.delete(self.agent)
        initial_state = self.env.initialState * UNIT
        self.agent = self.canvas.create_rectangle(initial_state[1] + UNIT / 3,
                                                  initial_state[0] + UNIT / 3,
                                                  initial_state[1] + UNIT - UNIT / 3,
                                                  initial_state[0] + UNIT - UNIT / 3,
                                                  fill='blue')
        self.canvas.tag_raise(self.agent)
        time.sleep(0.001)
        self.update()


# Plotting results functions
def plot_converged_solution(exp_n, tsteps):
    plt.grid(True)
    plt.title('Converged solution in number of time-steps')
    plt.ylabel('Time-steps')
    plt.xlabel('Experiment number')
    plt.plot(exp_n, tsteps, 'bo')
    plt.pause(0.001)


def plot_episode_solution(epis_n, tsteps):
    plt.grid(True)
    plt.title('Convergence')
    plt.ylabel('Time-steps')
    plt.xlabel('Episode number')
    plt.plot(epis_n, tsteps, 'bo')
    plt.pause(0.001)


# State Normalization
def normalize(state):
    return [(9-state[0])/9, (9-state[1])/9]
