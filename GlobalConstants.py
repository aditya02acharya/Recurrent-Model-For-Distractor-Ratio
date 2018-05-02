N_ROWS = 6
KERNEL_SIZE = 3
PEX = 1.8
FEATURE_SIZE = 1.3
KERNEL_SIZE = 3
SEED = 123
N_DR_ELEMENTS = N_ROWS * N_ROWS
MAX_ACTIONS = N_DR_ELEMENTS + 2
PRESENT = N_DR_ELEMENTS + 1
ABSENT = N_DR_ELEMENTS
MAX_STEPS = 15
LEARNING_RATE = 0.001

NETWORK_INPUT = N_DR_ELEMENTS
NETWORK_OUTPUT = MAX_ACTIONS

TRUE = 1
FALSE = 0
SPATIAL_NOISE_COLOUR = 6.0
SPATIAL_NOISE_SHAPE = 6.0
FEATURE_NOISE_COLOUR = 17.0
FEATURE_NOISE_SHAPE = 12.0
CONST_VAR = 0.0001
STEP_COST = -0.1
CORRECT_REWARD = 1
WRONG_REWARD = -1
FEATURE_COLOUR = 'colour'
FEATURE_SHAPE = 'shape'
PARAMETER_V = 0.7
PARAMETER_B = 0.1
PARAMETER_C = 0.1
PARAMETER_A_COL = 0.035
PARAMETER_A_SHP = 0.3
RATIO = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]
