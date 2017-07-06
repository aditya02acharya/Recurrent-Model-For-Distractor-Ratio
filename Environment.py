from rl.core import Env
import numpy as np
from math import sqrt
from scipy.spatial import distance
import random
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
        self.history_col = np.zeros((N_ROWS, N_ROWS))
        self.history_shp = np.zeros((N_ROWS, N_ROWS))
	self.action_input = np.reshape(np.zeros((N_ROWS, N_ROWS)).flatten(), [36])
	self.colour = np.reshape(np.zeros((N_ROWS, N_ROWS)).flatten(), [36])
	self.shape = np.reshape(np.zeros((N_ROWS, N_ROWS)).flatten(), [36])
	self.focus = None

    def kalman_update(self, estimate_val, estimate_sd, new_val, new_sd):

        # Calculate Kalman gain, and use square of sd to convert it to variance
        kalman_gain = (estimate_sd*estimate_sd)/((estimate_sd*estimate_sd) + (new_sd*new_sd))

        update_val = estimate_val + kalman_gain * (new_val - estimate_val)

        update_sd = sqrt((1 - kalman_gain) * (estimate_sd * estimate_sd))

        return update_val, update_sd
	
    def relevance_update(self, colour_estimate, colour_sd, shape_estimate, shape_sd):
	kalman_gain = (colour_sd*colour_sd)/((colour_sd*colour_sd)+(shape_sd*shape_sd))
	update_val = sqrt(((1-kalman_gain)*((1-colour_estimate)**2)) + ((kalman_gain)*((1-shape_estimate)**2)))
	return update_val 
	 
    def step(self, action):
        self.steps += 1
        done = False
	obs_space_col = None
	obs_space_shp = None
	
        if action < N_DR_ELEMENTS:
	    
            obs_space_col, obs_space_shp  = self.model.sample(action, self.current_display, self.global_var)
	    self.action_input = np.zeros((N_ROWS, N_ROWS))
	    self.action_input[(action/N_ROWS)][(action%N_ROWS)] = 1
	    self.colour = obs_space_col
	    self.shape = obs_space_shp    


        if action < N_DR_ELEMENTS:
            reward = STEP_COST
        elif (action == PRESENT and self.current_display.get_target_status() == TRUE) or (
                        action == ABSENT and self.current_display.get_target_status() == FALSE):
            reward = CORRECT_REWARD
        else:
            reward = WRONG_REWARD

        if action == PRESENT or action == ABSENT or self.steps >= MAX_STEPS:
            done = True

        #observation = [self.history_col.flatten(), self.history_shp.flatten(), self.action_input.flatten()]
        #observation = np.asarray(temp).flatten()#np.multiply(self.history_col.flatten(), self.history_shp.flatten())#np.asarray(temp).flatten()
	#observation = [self.belief, self.action_input.flatten()]
	#f = np.zeros((N_ROWS, N_ROWS))
	#for row in range(0, N_ROWS):
	#	for col in range(0, N_ROWS):
	#			f[row][col] = distance.euclidean([1,1],[self.colour[row][col],self.shape[row][col]])
			
	obs_c = np.reshape(self.colour.flatten(), [36])
	obs_s = np.reshape(self.shape.flatten(), [36])
	focus = np.reshape(self.action_input.flatten(), [1,36])
	#focus = np.reshape(self.action_input.flatten(), [36])
	#obs = np.reshape(f.flatten(), [36])
	return obs_c, obs_s, float(reward), done
	#return obs, float(reward), done

    def reset(self):
        self.steps = 0
        self.current_display = self.generator.generate_random_display()
	self.action_input = np.reshape(np.zeros((N_ROWS, N_ROWS)).flatten(), [1,36])
        self.colour = np.reshape(np.zeros((N_ROWS, N_ROWS)).flatten(), [36])
        self.shape = np.reshape(np.zeros((N_ROWS, N_ROWS)).flatten(), [36])
        #observation = [self.history_col.flatten(), self.history_shp.flatten(), self.action_input.flatten()]
        #observation = np.asarray(temp).flatten()#np.multiply(self.history_col.flatten(), self.history_shp.flatten())#np.asarray(temp).flatten()
	#observation = [self.belief, self.action_input.flatten()]
	obs_c = self.colour
	obs_s = self.shape
	focus = self.action_input
        return obs_c, obs_s
	#return obs_c

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

#print obs.shape
#display = env.get_display()

#print "Colour : "
#print display.get_colour()

#print "Shape : "
#print display.get_shape()

#for ep in range(0, 10, 1):	
#	print obs
#	action = random.randint(0,N_DR_ELEMENTS)
#	print "Action : " + str(action)
#	obs, r, done = env.step(action)

