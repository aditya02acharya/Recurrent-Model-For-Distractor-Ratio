from GlobalConstants import *
import numpy as np
from scipy.spatial import distance
from math import sqrt
from math import exp
from math import pi
from GlobalVariables import GlobalVariables
from DisplayGenerator import DisplayGenerator 

np.random.seed(123)

class ObservationModel(object):

    def sample(self, action, current_display, global_variables):
        """
        Samples a random observation from a given display.
        :param seed:
        :return: 2D array with colour and shape noisy observation
        """
        x = int(action / N_ROWS)
        y = int(action % N_ROWS)
        obs_space_colour = np.ones((N_ROWS, N_ROWS))*0.5
        obs_space_shape = np.ones((N_ROWS, N_ROWS))*0.5
        i_coords, j_coords = np.meshgrid(range(N_ROWS), range(N_ROWS), indexing='ij')	
        coords = np.dstack((i_coords, j_coords))

        dist_matrix = distance.cdist(coords.reshape(-1, 2), [[x,y]]).reshape(coords.shape[:2]) * PEX

        obs_space_colour = self.observe_feature(current_display.get_colour(), obs_space_colour, dist_matrix, global_variables, "COLOUR")
        obs_space_shape = self.observe_feature(current_display.get_shape(), obs_space_shape, dist_matrix, global_variables, "SHAPE")

        return obs_space_colour,obs_space_shape,dist_matrix

    def observe_feature(self, features, obs_space, dist_matrix, global_variables, feature_type):

        #temp = self.add_spatial_noise(features, obs_space, dist_matrix, global_variables, feature_type)
        obs_space = self.add_keras_feature_noise(features, obs_space, dist_matrix, global_variables, feature_type)
	
        return obs_space
	
    def add_keras_feature_noise(self, features, obs_space, dist_matrix, global_variables, feature_type):

        X = np.random.normal(FEATURE_SIZE, 0.7*FEATURE_SIZE, N_ROWS*N_ROWS)
        indexes = np.arange(N_ROWS*N_ROWS)
        
        if feature_type == "COLOUR":
            threshold = (0.035 * (dist_matrix**2)) + (0.1 * dist_matrix) + 0.1
        else:
            threshold = (0.3 * (dist_matrix**2)) + (0.1 * dist_matrix) + 0.1

        win_ind = indexes[(FEATURE_SIZE + X) > threshold.ravel()]
        obs_space.ravel()[win_ind] = features.ravel()[win_ind]	

        return obs_space
	

    def add_feature_noise(self, features, obs_space, x, y, global_variables, feature_type):
       
        for ext_x in range(0, N_ROWS, 1):
            for ext_y in range(0, N_ROWS, 1):
                #obs_space[ext_x,ext_y,2] = global_variables.get_eccentricity(x, y, ext_x, ext_y)
                if feature_type == "COLOUR":
                    obs_space[ext_x,ext_y,0] = features[ext_x,ext_y,0] + np.random.normal(0, global_variables.get_feature_noise_colour_sd(x, y, ext_x, ext_y), 1)[0]
                else:
                    obs_space[ext_x,ext_y,1] = features[ext_x,ext_y,1] + np.random.normal(0, global_variables.get_feature_noise_shape_sd(x, y, ext_x, ext_y), 1)[0]

        return obs_space

    def add_spatial_noise(self, features, obs_space, x, y, global_variables, feature_type):
        for ext_x in range(0, N_ROWS, 1):
            for ext_y in range(0, N_ROWS, 1):
                if feature_type == "COLOUR":
                    sigma = global_variables.get_spatial_noise_colour_sd(x, y, ext_x, ext_y)
                    kernel = self.generate_gaussian_kernel(sigma)
                    obs_space[ext_x,ext_y,0] = self.single_value_convolution(features, ext_x, ext_y, kernel)
                else:
                    sigma = global_variables.get_spatial_noise_shape_sd(x, y, ext_x, ext_y)
                    kernel = self.generate_gaussian_kernel(sigma)
                    obs_space[ext_x,ext_y,1] = self.single_value_convolution(features, ext_x, ext_y, kernel)


        return obs_space

    def generate_gaussian_kernel(self, sigma):
        """
        Function to generate gaussian kernel.

        :param sigma:
        :return: 2D matrix of KERNEL_SIZE * KERNEL_SIZE
        """

        kernel = np.zeros((KERNEL_SIZE, KERNEL_SIZE))

        s = 2.0 * sigma * sigma

        size = int((KERNEL_SIZE - 1)/2)

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

        size = int((KERNEL_SIZE-1)/2)

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


#######TEST########
#g_var = GlobalVariables()

#generator = DisplayGenerator()

#disp = generator.generate_random_display()

#print(disp.get_colour())

#print(disp.get_shape())

#model = ObservationModel()

#obs_col, obs_shp, ecc = model.sample(8, disp, g_var)

#print(obs_col)
#print(obs_shp)
#print(ecc)