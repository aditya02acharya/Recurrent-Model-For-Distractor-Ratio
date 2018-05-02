from rl.core import Env
import numpy as np
from math import sqrt
from scipy.spatial import distance
from Actions import Actions
from GlobalConstants import *
from GlobalVariables import GlobalVariables
from DisplayGenerator import DisplayGenerator
from ObservationModel import ObservationModel


class Environment(Env):

    def __init__(self):
        self.steps = 0
        Env.action_space = Actions()
        Env.reward_range = (WRONG_REWARD, CORRECT_REWARD)
        self.global_var = GlobalVariables()
        self.generator = DisplayGenerator()
        self.model = ObservationModel()
        self.current_display = None
        self.history = np.ones((N_ROWS, N_ROWS, 2)) * 0.5

    def kalman_update(self, estimate_val, estimate_sd, new_val, new_sd):

        # Calculate Kalman gain, and use square of sd to convert it to variance
        kalman_gain = (estimate_sd*estimate_sd)/((estimate_sd*estimate_sd) + (new_sd*new_sd))

        update_val = estimate_val + kalman_gain * (new_val - estimate_val)

        update_sd = sqrt((1 - kalman_gain) * (estimate_sd * estimate_sd))

        return update_val, update_sd

    def step(self, action):
        self.steps += 1
        done = False

        if action < N_DR_ELEMENTS:
            obs_space = self.model.sample(action, self.current_display, self.global_var)
	    self.history = obs_space

        if action < N_DR_ELEMENTS:
            reward = STEP_COST
        elif (action == PRESENT and self.current_display.get_target_status() == TRUE) or (
                        action == ABSENT and self.current_display.get_target_status() == FALSE):
            reward = CORRECT_REWARD + (STEP_COST * (self.steps - 1))
        else:
            reward = WRONG_REWARD + (STEP_COST * (self.steps - 1))

        if action == PRESENT or action == ABSENT or self.steps >= MAX_STEPS:
            done = True

        return self.history.flatten(), float(reward), done

    def reset(self):
        self.steps = 0
        self.current_display = self.generator.generate_random_display()
        self.history = np.ones((N_ROWS, N_ROWS, 2)) * 0.5
        return self.history.flatten()

    def render(self, mode='human', close=False):
        print "Render function"

    def close(self):
        print "close function"

    def seed(self, seed=None):
        print "seed function"

    def configure(self, *args, **kwargs):
        print "configure function"

    def get_display(self):
	return self.current_display


#env = Environment()

#obs = env.reset()

#print env.get_display().get_colour()
#print " "
#print env.get_display().get_shape()

#obs, r, d = env.step(8)

#print obs

