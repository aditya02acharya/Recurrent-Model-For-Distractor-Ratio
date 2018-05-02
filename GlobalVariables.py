from GlobalConstants import *
from scipy.spatial import distance

class GlobalVariables(object):

    def __init__(self):
        self.distances = {}
        self.sdFeatureNoiseColour = {}
        self.sdFeatureNoiseShape = {}
        self.sdSpatialNoiseColour = {}
        self.sdSpatialNoiseShape = {}

        for fix_x in range(0, N_ROWS, 1):
            for fix_y in range(0, N_ROWS, 1):
                for ext_x in range(0, N_ROWS, 1):
                    for ext_y in range(0, N_ROWS, 1):
                        key = str(fix_x) + "," + str(fix_y) + "," + str(ext_x) + "," + str(ext_y)
                        dist = distance.euclidean([fix_x, fix_y], [ext_x, ext_y]) * PEX
                        self.distances.update({key: dist})
                        self.sdSpatialNoiseColour.update({key: (dist / SPATIAL_NOISE_COLOUR + CONST_VAR)})
			self.sdSpatialNoiseShape.update({key: (dist / SPATIAL_NOISE_SHAPE + CONST_VAR)})
                        self.sdFeatureNoiseColour.update({key: (dist / FEATURE_NOISE_COLOUR + CONST_VAR)})
			self.sdFeatureNoiseShape.update({key: (dist / FEATURE_NOISE_SHAPE + CONST_VAR)})


    def get_key(self, fix_x, fix_y, ext_x, ext_y):

        key = str(fix_x) + "," + str(fix_y) + "," + str(ext_x) + "," + str(ext_y)
        return key

    def get_eccentricity(self, fix_x, fix_y, ext_x, ext_y):
        key = self.get_key(fix_x, fix_y, ext_x, ext_y)

        return self.distances.get(key)

    def get_feature_noise_colour_sd(self, fix_x, fix_y, ext_x, ext_y):
        key = self.get_key(fix_x, fix_y, ext_x, ext_y)

        return self.sdFeatureNoiseColour.get(key)

    def get_feature_noise_shape_sd(self, fix_x, fix_y, ext_x, ext_y):
        key = self.get_key(fix_x, fix_y, ext_x, ext_y)

        return self.sdFeatureNoiseShape.get(key)

    def get_spatial_noise_colour_sd(self, fix_x, fix_y, ext_x, ext_y):
        key = self.get_key(fix_x, fix_y, ext_x, ext_y)

        return self.sdSpatialNoiseColour.get(key)

    def get_spatial_noise_shape_sd(self, fix_x, fix_y, ext_x, ext_y):
        key = self.get_key(fix_x, fix_y, ext_x, ext_y)

        return self.sdSpatialNoiseShape.get(key)
