import numpy as np
import matplotlib.pyplot as plt

from distributions import onepoint, zero, gaussian
from system import Plant, Channel, LQGCost
from simulation import simulate
from coding import TrivialEncoder, TrivialDecoder


# Time horizon
T = 100

# System coefficient
alpha = 2
assert(alpha > 1) # unstable

# Variance (noise power)
P0 = 100 # V[x0]
W  = 1 # V[w(t)]
V  = 1 # V[v(t)]
SNR = 5 # 1 / V[n(t)]


plant = Plant(alpha, gaussian(P0), gaussian(W), gaussian(V))
channel = Channel(gaussian(1 / SNR))
encoder = TrivialEncoder()
decoder = TrivialDecoder()

Q = 1
R = 1
F = 1
LQG = LQGCost(plant, Q, R, F)

LQG_trajectory = [LQG.evaluate(t)
        for t in simulate(plant, channel, encoder, decoder, LQG, T + 1)]

plt.plot(range(0, T + 1), LQG_trajectory)
plt.show()
