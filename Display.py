from GlobalConstants import *
import numpy as np

class Display(object):

    def __init__(self):
        self.colour = np.zeros((N_ROWS, N_ROWS))
        self.shape = np.zeros((N_ROWS, N_ROWS))
        self.targetStatus = FALSE
        self.sameColourRatio = -1
        self.targetX = 0
        self.targetY = 0

    def get_colour(self):
        return self.colour

    def set_colour(self, colour):
        self.colour = colour

    def get_shape(self):
        return self.shape

    def set_shape(self, shape):
        self.shape = shape

    def get_target_status(self):
        return self.targetStatus

    def set_target_status(self, status):
        self.targetStatus = status

    def get_same_colour_ratio(self):
        return self.sameColourRatio

    def set_same_colour_ratio(self, ratio):
        self.sameColourRatio = ratio

    def get_target_x(self):
        return self.targetX

    def get_target_y(self):
        return self.targetY

    def set_target(self, x, y):
        self.targetX = x
        self.targetY = y
        self.colour[x][y] = 1
        self.shape[x][y] = 1

    def add_new_display(self, feature_matrix):
        indexes = np.nonzero(feature_matrix.ravel())
        self.colour.ravel()[indexes] = 1
        ind = np.where(feature_matrix.ravel() == 0)
        self.shape.ravel()[ind] = 1
