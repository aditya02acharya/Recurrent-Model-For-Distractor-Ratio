from gym import spaces
from rl.core import Space
from GlobalConstants import *

class Actions(Space):

    def __init__(self):
        self.action_space = spaces.Discrete(MAX_ACTIONS)

    def sample(self, seed=None):
        return self.action_space.sample()

    def contains(self, x):
        """
        Return boolean specifying if x is a valid member of this space
        :rtype: int
        """
        return self.action_space.contains(x)
