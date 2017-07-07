from utilities import int_binsearch

import numpy as np
from scipy.integrate import quad

class Encoder:
    def __init__(self, boundaries):
        self.boundaries = boundaries

    def encode(self, *msg):
        assert(len(msg) == 1) # One real number
        x = msg[0]
        return int_binsearch(lambda i: self.boundaries[i] >= x,
                0, len(self.boundaries)) - 1


class Decoder:
    def __init__(self, levels):
        self.levels = levels

    def decode(self, *msg):
        assert(len(msg) == 1) # One integer
        i = msg[0]
        return self.levels[i]


def generate_intervals(n_levels, distr):
    # Initialize the boundaries (TODO: can be done more efficiently)
    # Start with the same total probability 1/n_levels in each interval
    boundaries = distr.ppf(np.linspace(0, 1, n_levels + 1))

    def boundaries_to_levels(boundaries):
        return [
                quad(lambda x: x * distr.pdf(x), lo, hi)[0]
              / (distr.cdf(hi) - distr.cdf(lo))
            for lo, hi in zip(boundaries, boundaries[1:])]

    def levels_to_boundaries(levels):
        return ([-float('inf')]
              + [(a + b) / 2 for a, b in zip(levels, levels[1:])]
              + [float('inf')])

    for _ in range(10): # TODO more sensible termination condition
        levels = boundaries_to_levels(boundaries)
        boundaries = levels_to_boundaries(levels)

    return levels, boundaries
