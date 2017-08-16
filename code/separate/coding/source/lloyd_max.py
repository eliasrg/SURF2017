from utilities import int_binsearch

import numpy as np
from scipy.integrate import quad
from numpy import linalg as LA

# Integral subdivision limit
LIMIT = 10

# Maximum number of iterations for the Lloyd-Max algorithm
MAX_ITERATIONS = 200

class Encoder:
    def __init__(self, boundaries):
        self.boundaries = boundaries

    def encode(self, *msg):
        assert len(msg) == 1 # One real number
        x = msg[0]
        return int_binsearch(lambda i: self.boundaries[i] >= x,
                0, len(self.boundaries)) - 1


class Decoder:
    def __init__(self, levels, boundaries):
        self.levels = levels
        self.boundaries = boundaries

    def decode(self, *msg):
        assert len(msg) == 1 # One integer
        i = msg[0]
        return self.levels[i]

    def get_interval(self, i):
        return self.boundaries[i], self.boundaries[i + 1]


def generate(n_levels, distr):
    # Initialize the boundaries (TODO: can be done more efficiently)
    # Start with the same total probability 1/n_levels in each interval
    boundaries = distr.ppf(np.linspace(0, 1, n_levels + 1))

    def boundaries_to_levels(boundaries):
        return [
                quad(lambda x: x * distr.pdf(x), lo, hi, limit=LIMIT)[0]
              / quad(distr.pdf, lo, hi, limit=LIMIT)[0]
            for lo, hi in zip(boundaries, boundaries[1:])]

    def levels_to_boundaries(levels):
        return ([-float('inf')]
              + [(a + b) / 2 for a, b in zip(levels, levels[1:])]
              + [float('inf')])

    for i in range(MAX_ITERATIONS):
        prev_boundaries = boundaries
        levels = boundaries_to_levels(boundaries)
        boundaries = levels_to_boundaries(levels)

        # Normalized Euclidean distance between previous boundaries and
        # next boundaries
        dist = LA.norm(np.array(boundaries[1:-1]) \
                - np.array(prev_boundaries[1:-1])) / np.sqrt(n_levels)
        if dist < 0.001:
            break
    else:
        raise RuntimeError("Lloyd-Max failed to converge after {} iterations."
                .format(MAX_ITERATIONS))

    return Encoder(boundaries), Decoder(levels, boundaries)
