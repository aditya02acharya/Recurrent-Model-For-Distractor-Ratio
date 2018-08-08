import numpy as np
from math import sqrt
from scipy.spatial import distance
from GlobalConstants import *
from GlobalVariables import GlobalVariables
from DisplayGenerator import DisplayGenerator
from ObservationModel import ObservationModel


class Environment():

    def __init__(self):
        self.steps = 0
        self.global_var = GlobalVariables()
        self.generator = DisplayGenerator()
        self.model = ObservationModel()
        self.current_display = None
        self.correct = 0
        self.prev_action = -1
        self.total_time = 0.0
        self.history = None


    def step(self, action):
        self.steps += 1
        done = False

        if action < N_DR_ELEMENTS:
            obs_space_colour, obs_space_shape, eccentricity = self.model.sample(action, self.current_display, self.global_var)
            self.history = np.dstack((obs_space_colour, obs_space_shape))
            self.focus = np.eye(N_DR_ELEMENTS)[action]

        if action < N_DR_ELEMENTS:
            if not self.prev_action == -1:
                e = distance.euclidean([int(self.prev_action/N_ROWS), int(self.prev_action%N_ROWS)], 
                                        [int(action/N_ROWS), int(action%N_ROWS)]) * PEX
                if self.current_display.get_target_status() == 1:
                    reward = (STEP_COST_PRESENT + ((37.0 + 2.7 * e)/1000.0)) * -1
                    self.total_time += (STEP_COST_PRESENT + ((37.0 + 2.7 * e)/1000.0))
                else:
                    reward = (STEP_COST_ABSENT + ((37.0 + 2.7 * e)/1000.0)) * -1
                    self.total_time += (STEP_COST_ABSENT + ((37.0 + 2.7 * e)/1000.0))
            else:
                if self.current_display.get_target_status() == 1:
                    reward = (STEP_COST_PRESENT) * -1
                    self.total_time += (STEP_COST_PRESENT)
                else:
                    reward = (STEP_COST_ABSENT ) * -1
                    self.total_time += (STEP_COST_ABSENT)
                
        elif (action == PRESENT and self.current_display.get_target_status() == TRUE) or (
                        action == ABSENT and self.current_display.get_target_status() == FALSE):
            reward = CORRECT_REWARD
            self.correct = 1
        else:
            reward = WRONG_REWARD

        if action == PRESENT or action == ABSENT:
            done = True
        if self.steps >= MAX_STEPS:
            done = True
            reward = WRONG_REWARD
        self.prev_action = action
        return self.history.flatten(), self.focus.flatten(), float(reward), done

    def reset(self):
        self.steps = 0
        self.correct = 0
        self.total_time = 0.0
        self.prev_action = -1
        self.current_display = self.generator.generate_random_display()
        self.history = np.ones((N_ROWS, N_ROWS,2)) * 0.5
        #self.ecc = np.zeros((N_ROWS, N_ROWS))
        #self.history = np.dstack((self.obs,self.ecc))
        self.focus = np.zeros((N_ROWS, N_ROWS))

        return self.history.flatten(), self.focus.flatten()

    def render(self, mode='human', close=False):
        print("Render function")

    def close(self):
        print("close function")

    def seed(self, seed=None):
        print ("seed function")

    def configure(self, *args, **kwargs):
        print ("configure function")

    def get_display(self):
        return self.current_display


env = Environment()

obs = env.reset()

print(env.get_display().get_colour())
print(" ")
print(env.get_display().get_shape())

obs,f, r, d = env.step(8)

print(obs)
print(f)

