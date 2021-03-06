from GlobalConstants import *
import numpy as np
import timeit
from random import shuffle
from random import randint
from Display import Display

class DisplayGenerator(object):

    def __init__(self):

        self.sample_list = list(range(0, N_DR_ELEMENTS, 1))
        

    def generate_random_display(self):

        display = Display()

        features = np.zeros((N_ROWS, N_ROWS))

        #randomly choose target status for a given display. Either present : 1 or absent : 0
        target_status = randint(0,1)

        #randomly choose the distractor ratio for the given
        same_colour_ratio = RATIO[randint(0,len(RATIO)-1)]

        random_list = np.random.choice(self.sample_list, same_colour_ratio, replace=False)
	
        features.ravel()[random_list] = TRUE

        #initialize new Display
        display.add_new_display(features)
        display.set_same_colour_ratio(same_colour_ratio)
        display.set_target_status(target_status)

        #randomly choose a location to place the target in the display if the target is present
        if target_status == TRUE:
            loc = randint(0,N_DR_ELEMENTS-1)
            display.set_target(int(loc/N_ROWS), int(loc%N_ROWS))

        return display






    def reservoir_sampling(self, samples, sample_list):

        random_list = np.zeros(samples)

        count = 0

        for item in sample_list:
            if count < samples:
                random_list[count] = item
            else:
                randomPos = randint(0, count)
                if randomPos < samples:
                    random_list[randomPos] = item

            count = count + 1

        return random_list

#gen = DisplayGenerator()
#display = gen.generate_random_display()
#print(display.get_colour())

#print(display.get_shape())
