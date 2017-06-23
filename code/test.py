import numpy as np
import matplotlib.pyplot as plt
import itertools as it

from misc import onepoint, zero, gaussian
from system import Plant, Channel, LQGCost
from simulation import simulate
from coding import TrivialEncoder, TrivialDecoder


# Variance (noise power)
P0 = 100 # V[x0]
W  = 1 # V[w(t)]
V  = 1 # V[v(t)]


plant = Plant(2, gaussian(P0), gaussian(W), gaussian(V))
# plant = Plant(2, onepoint(10000), zero, zero)
channel = Channel(zero)
encoder = TrivialEncoder()
decoder = TrivialDecoder()

Q = 1
R = 1
F = 1
LQG = LQGCost(plant, Q, R, F)

LQG_trajectory = [LQG.evaluate()
        for _ in it.islice(simulate(plant, channel, encoder, decoder, LQG), 40)]

plt.plot(LQG_trajectory)
plt.show()
