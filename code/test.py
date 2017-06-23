import numpy as np
import matplotlib.pyplot as plt
import itertools as it

from distributions import onepoint, zero, gaussian
from system import Plant, Channel, LQGCost
from simulation import simulate
from coding import TrivialEncoder, TrivialDecoder


# System coefficient
a = 2
assert(a > 1) # unstable

# Variance (noise power)
P0 = 100 # V[x0]
W  = 1 # V[w(t)]
V  = 1 # V[v(t)]
SNR = 5 # 1 / V[n(t)]


plant = Plant(a, gaussian(P0), gaussian(W), gaussian(V))
channel = Channel(gaussian(1 / SNR))
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
