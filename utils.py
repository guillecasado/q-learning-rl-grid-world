import numpy as np
import tkinter as tk
import time
import matplotlib.pyplot as plt
import environment

environment.UNIT = 50  # pixels per environment.UNIT
GRID_HEIGHT = 10
GRID_WIDTH = 10
ACTIONS = np.array([0, 1, 2, 3])  # up, down, left, right
COORD_ACTIONS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])  # up, down, left, right
WALLS = np.array(
    [(0, 2), (0, 5), (1, 2), (1, 5), (2, 2), (2, 7), (2, 8), (2, 9), (3, 2), (4, 2), (4, 5), (5, 5), (6, 0),
     (6, 8), (6, 9), (7, 2), (7, 3), (7, 4), (7, 5), (9, 3), (7, 6)]
)


# Grid world real-time display class
class GraphicDisplay(tk.Tk):
    def __init__(self, env):
        tk.Tk.__init__(self)
        self.env = env
        self.title('Environment')
        self.geometry('{0}x{1}'.format(self.env.width * environment.UNIT, self.env.height * environment.UNIT))
        self.resizable(False, False)
        self.canvas, self.agent = self.create_canvas()
        self.texts = []
        self.coordActions = environment.COORD_ACTIONS

    def create_canvas(self):
        # Initialize the Canvas
        canvas = tk.Canvas(self, bg='white', highlightthickness=0.1, highlightbackground="black",
                           height=self.env.height * environment.UNIT,
                           width=self.env.width * environment.UNIT)

        # Create Grids
        for col in range(0, self.env.width * environment.UNIT, environment.UNIT):
            x0, y0, x1, y1 = col, 0, col, self.env.height*environment.UNIT
            canvas.create_line(x0, y0, x1, y1)

        for row in range(0, self.env.height * environment.UNIT, environment.UNIT):
            x0, y0, x1, y1 = 0, row, self.env.width*environment.UNIT, row
            canvas.create_line(x0, y0, x1, y1)

        # Create Walls
        for wall in self.env.walls:
            wall_pos = wall * environment.UNIT
            canvas.create_rectangle(wall_pos[1], wall_pos[0],
                                    wall_pos[1]+environment.UNIT, wall_pos[0]+environment.UNIT,
                                    fill='dimgray')

        # Create Initial State
        initial_state = self.env.initialState * environment.UNIT
        agent = canvas.create_rectangle(initial_state[1] + environment.UNIT/3,
                                        initial_state[0] + environment.UNIT/3,
                                        initial_state[1] + environment.UNIT - environment.UNIT/3,
                                        initial_state[0] + environment.UNIT - environment.UNIT/3,
                                        fill='blue')

        # Create Goal State
        goal_state = self.env.goalState * environment.UNIT
        canvas.create_rectangle(goal_state[1], goal_state[0],
                                goal_state[1] + environment.UNIT, goal_state[0] + environment.UNIT, fill='green')
        canvas.pack()
        return canvas, agent

    def text_q_value(self, row, col, value, action, font='Helvetica', size=7, style='normal', anchor="nw"):
        if (sum([((row, col) == wall).all() for wall in self.env.walls])) == 0 \
                and ((row, col) != self.env.goalState).any():
            text_position = np.array([(environment.UNIT*(col+1/2.4), environment.UNIT*(row+1/20)),
                                      (environment.UNIT*(col+1/2.4), environment.UNIT*(row+1/1.25)),
                                      (environment.UNIT*(col+1/20), environment.UNIT*(row+1/2.5)),
                                      (environment.UNIT*(col+1/1.4), environment.UNIT*(row+1/2.55))])
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

    def step(self, action, q_table):
        agent_move = [self.coordActions[action][1]*environment.UNIT, self.coordActions[action][0] * environment.UNIT]
        self.canvas.move(self.agent, agent_move[0], agent_move[1])
        self.print_q_values(q_table)
        self.canvas.tag_raise(self.agent)
        time.sleep(0.001)
        self.update()

    def reset_display(self):
        self.canvas.delete(self.agent)
        initial_state = self.env.initialState * environment.UNIT
        self.agent = self.canvas.create_rectangle(initial_state[1] + environment.UNIT / 3,
                                                  initial_state[0] + environment.UNIT / 3,
                                                  initial_state[1] + environment.UNIT - environment.UNIT / 3,
                                                  initial_state[0] + environment.UNIT - environment.UNIT / 3,
                                                  fill='blue')
        self.canvas.tag_raise(self.agent)
        time.sleep(0.001)
        self.update()


# Plotting results functions
def time_steps_converged_solution_vs_experiment(exp_n, tsteps):
    # plt.plot((data_list[i-1][0], data_list[i][0]), (data_list[i-1][1], data_list[i][1]), 'ro-')
    plt.plot(exp_n, tsteps, 'bo')
    plt.pause(0.01)
