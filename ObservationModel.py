from GlobalConstants import *
from DisplayGenerator import DisplayGenerator
from GlobalVariables import GlobalVariables
import numpy as np
from math import sqrt
from math import exp
from math import pi

np.set_printoptions(suppress=True)

class ObservationModel(object):

    def sample(self, action, current_display, global_variables):
        """
        Samples a random observation from a given display.
        :param seed:
        :return: 2D array with colour and shape noisy observation
        """
        x = action / N_ROWS
        y = action % N_ROWS
        obs_space_col = self.observe_feature(current_display.get_colour(), x, y, global_variables, FEATURE_COLOUR)
        obs_space_shp = self.observe_feature(current_display.get_shape(), x, y, global_variables, FEATURE_SHAPE)
        
	return obs_space_col, obs_space_shp

    def observe_feature(self, features, x, y, global_variables, feature):

	observation = self.add_keras_feature_noise(features, x, y, global_variables, feature)
        
	#temp = self.add_feature_noise(features, x, y, global_variables, feature)

       	#observation = self.add_spatial_noise(temp, x, y, global_variables, feature)

        return observation

    def add_feature_noise(self, obs, x, y, global_variables, feature):
	obs_space = np.zeros((N_ROWS, N_ROWS))
	for ext_x in range(0, N_ROWS, 1):
		for ext_y in range(0, N_ROWS, 1):
			obs_space[ext_x][ext_y] = obs[ext_x][ext_y] + np.random.normal(0, global_variables.get_feature_noise_sd(x, y, ext_x, ext_y, feature), 1)[0]
	
	return obs_space	
	

    def add_keras_feature_noise(self, obs, x, y, global_variables, feature):
        obs_space = np.ones((N_ROWS, N_ROWS)) * -1
        for ext_x in range(0, N_ROWS, 1):
            for ext_y in range(0, N_ROWS, 1):
                
		e = global_variables.get_eccentricity(x, y, ext_x, ext_y)	
		
		X = np.random.normal(FEATURE_SIZE, 0.7*FEATURE_SIZE)

		if feature == FEATURE_COLOUR:
			threshold = (0.035 * e * e) + (0.1 * e) + 0.1
			if (FEATURE_SIZE + X) > threshold:
				obs_space[ext_x][ext_y] = obs[ext_x][ext_y]
			else:
				obs_space[ext_x][ext_y] = 0
		else:
			threshold = (0.3 * e * e) + (0.1 * e) + 0.1
			if (FEATURE_SIZE + X) > threshold:
				obs_space[ext_x][ext_y] = obs[ext_x][ext_y]
			else:
				obs_space[ext_x][ext_y] = 0

				
        return obs_space

    def add_spatial_noise(self, obs, x, y, global_variables, feature):
        obs_space = np.zeros((N_ROWS, N_ROWS))
        for ext_x in range(0, N_ROWS, 1):
            for ext_y in range(0, N_ROWS, 1):
                sigma = global_variables.get_spatial_noise_sd(x, y, ext_x, ext_y, feature)
                kernel = self.generate_gaussian_kernel(sigma)
                obs_space[ext_x][ext_y] = self.single_value_convolution(obs, ext_x, ext_y, kernel)

        return obs_space

    def generate_gaussian_kernel(self, sigma):
        """
        Function to generate gaussian kernel.

        :param sigma:
        :return: 2D matrix of KERNEL_SIZE * KERNEL_SIZE
        """

        kernel = np.zeros((KERNEL_SIZE, KERNEL_SIZE))

        s = 2.0 * sigma * sigma

        size = (KERNEL_SIZE - 1)/2

        sum = 0.0

        for x in range(-size, size+1, 1):
            for y in range(-size, size+1, 1):
                radius = sqrt(x * x + y * y)
                kernel[x + size][y + size] = exp(-1 * ((radius * radius)/s))/(pi * s)
                sum = sum + kernel[x + size][y + size]

        #normalise kernel
        for row in range(0, KERNEL_SIZE, 1):
            for col in range(0, KERNEL_SIZE, 1):
                kernel[row][col] = kernel[row][col]/sum

        return kernel

    def single_value_convolution(self, features, x, y, kernel):

        output = 0.0

        size = (KERNEL_SIZE-1)/2

        temp = np.zeros((KERNEL_SIZE, KERNEL_SIZE))

        for i in range(-size, size+1, 1):
            for j in range(-size, size+1, 1):

                if (x + i > -1) and (y + j > -1) and (x + i < N_ROWS) and (y + j < N_ROWS):
                    temp[i+size][j+size] = features[x+i][y+j]
                else:
                    temp[i+size][j+size] = 0

        for row in range(0, KERNEL_SIZE, 1):
            for col in range(0, KERNEL_SIZE, 1):
                output = output + kernel[row][col] * temp[row][col]

        return output


gen = DisplayGenerator()

display = gen.generate_random_display()

print display.get_colour()
print " "
print display.get_shape()
print " "
print display.get_target_status()
var = GlobalVariables()

model = ObservationModel()

obs_col, obs_shp = model.sample(14, display, var)

print obs_col
print "  "
print obs_shp

