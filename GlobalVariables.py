from GlobalConstants import *
from scipy.spatial import distance

class GlobalVariables(object):

    def __init__(self):
        self.distances = {}
        self.sdColourFeatureNoise = {}
	self.sdShapeFeatureNoise = {}
        self.sdColourSpatialNoise = {}
	self.sdShapeSpatialNoise = {}

        for fix_x in range(0, N_ROWS, 1):
            for fix_y in range(0, N_ROWS, 1):
                for ext_x in range(0, N_ROWS, 1):
                    for ext_y in range(0, N_ROWS, 1):
                        key = str(fix_x) + "," + str(fix_y) + "," + str(ext_x) + "," + str(ext_y)
                        dist = distance.euclidean([fix_x, fix_y], [ext_x, ext_y]) * PEX
                        self.distances.update({key: round(dist,2)})
                        self.sdColourSpatialNoise.update({key: (dist / SPATIAL_NOISE_COLOUR + CONST_VAR)})
			self.sdShapeSpatialNoise.update({key: (dist / SPATIAL_NOISE_SHAPE + CONST_VAR)})
                        self.sdColourFeatureNoise.update({key: (dist / FEATURE_NOISE_COLOUR + CONST_VAR)})
			self.sdShapeFeatureNoise.update({key: (dist / FEATURE_NOISE_SHAPE + CONST_VAR)})


    def get_key(self, fix_x, fix_y, ext_x, ext_y):

        key = str(fix_x) + "," + str(fix_y) + "," + str(ext_x) + "," + str(ext_y)
        return key

    def get_eccentricity(self, fix_x, fix_y, ext_x, ext_y):
        key = self.get_key(fix_x, fix_y, ext_x, ext_y)

        return self.distances.get(key)

    def get_feature_noise_sd(self, fix_x, fix_y, ext_x, ext_y, feature):
        key = self.get_key(fix_x, fix_y, ext_x, ext_y)
	
	if feature == FEATURE_COLOUR:
		return self.sdColourFeatureNoise.get(key)
	else:
		return self.sdShapeFeatureNoise.get(key)	

    def get_spatial_noise_sd(self, fix_x, fix_y, ext_x, ext_y, feature):
        key = self.get_key(fix_x, fix_y, ext_x, ext_y)
	
	if feature == FEATURE_COLOUR:
		return self.sdColourSpatialNoise.get(key)
	else:
		return self.sdShapeSpatialNoise.get(key)

    def print_data(self):
        for ext_x in range(0, N_ROWS, 1):
	  for ext_y in range(0, N_ROWS, 1):
	    print self.get_eccentricity(0, 0, ext_x, ext_y)


