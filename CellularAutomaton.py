import pandas as pd
from scipy.interpolate import griddata, interp1d
from scipy.integrate import trapz, quad
import numpy as np
import time, csv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
import argparse


def anim_func(i, im, states):
    im.set_array(states[i].T)
    return im

class system_state():

    def __init__(self, dimensions, num_ones, seed = "", mutation_factor = 0.001):

        if seed:
            np.random.seed(seed)

        self.mutation_factor = mutation_factor
        self.dimensions = dimensions
        self.xmax = self.dimensions[0] -1
        self.ymax = self.dimensions[1] -1
        self.aspect_ratio = dimensions[0]/dimensions[1]
        self.state = np.zeros(shape = dimensions, dtype = np.bool_)
        self.state_len = int(dimensions[0]*dimensions[1])
        assert num_ones < self.state_len
        self.initial_num_ones = num_ones
        self.history = []
        self.timesteps = 0
        self.place_randoms(num_ones)


    def place_randoms(self, num_randoms):
        self.initial_positions = np.array([xy for xy in zip(np.random.randint(0, self.dimensions[0], int(self.initial_num_ones/2)),
                                                             np.random.randint(0, self.dimensions[1], int(self.initial_num_ones/2)))])
        for xy in self.initial_positions:
            self.state[xy[0], xy[1]] = 1

        self.history.append(self.state)
        return self.state


    def evolve_state(self):

        if self.timesteps > 0: self.history.append(self.state)
        self.new_state = np.zeros(shape = self.dimensions, dtype = np.bool_)
        for x in range(self.dimensions[0]):
            for y in range(self.dimensions[1]):
                self.new_state[x, y] = self.__rule_30(self.state[x, y], * self.get_lrtb(x, y))
                r = np.random.random(1)
                if (r[0] < self.mutation_factor):
                    self.new_state[x, y] = ~self.new_state[x, y]
        self.state = self.new_state
        self.timesteps += 1

        return self.state

    def __rule_0(self, mid, top, bottom, left, right):
        return left ^ (mid | right)

    def __rule_1(self, mid, top, bottom, left, right):
        return (bottom | top) & (left | right)

    def __rule_30(self, mid, top, bottom, left, right):
        return left ^ (mid | right) | bottom

    def get_lrtb(self, x, y):

        if (x >= self.xmax):
            left = self.state[self.xmax - 1, y]
            right = self.state[0, y]
        elif ((x > 0) & (x < self.xmax)):
            left = self.state[x - 1, y]
            right = self.state[x + 1, y]
        #If x is 0 then wrap around
        else:
            left = self.state[self.xmax, y]
            right = self.state[1, y]

        if (y >= self.ymax):
            bottom = self.state[x, self.ymax - 1]
            top = self.state[x, 0]
        elif ((y > 0) & (y < self.ymax)):
            bottom = self.state[x, y - 1]
            top = self.state[x, y + 1]
        else:
            bottom = self.state[x, self.ymax]
            top = self.state[x, 1]

        return top, bottom, right, left




state0 = system_state((100, 100), 200, seed = 1, mutation_factor = 0)
#state0.state = np.zeros(shape = (100, 100), dtype = np.bool_)
#state0.history = []
#state0.state[50, 50] = 1
#state0.history.append(state0.state)



fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(state0.state.T, origin = 'lower', interpolation = 'none', aspect = state0.aspect_ratio)
for i in range(100):
    state0.evolve_state()

anim = animation.FuncAnimation(fig, anim_func, frames = state0.timesteps, fargs = (im, state0.history, ), interval = 500)
anim.save("rule_30.mp4")
plt.show()
