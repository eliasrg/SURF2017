import numpy as np
import matplotlib.pyplot as plt

from distributions import onepoint, zero, gaussian
from system import Plant, Channel, LQGCost
from simulation import simulate, Parameters
from coding import TrivialEncoder, TrivialDecoder


params = Parameters(
        T = 100,
        alpha = 2,
        P0 = 100,
        W = 1, V = 1, SNR = 5,
        Q = 1, R = 1, F = 1)

locals().update(params.all()) # Bring parameters into scope


plant = Plant(alpha, gaussian(P0), gaussian(W), gaussian(V))
channel = Channel(gaussian(1 / SNR))
encoder = TrivialEncoder()
decoder = TrivialDecoder()

LQG = LQGCost(plant, Q, R, F)


LQG_trajectory = [LQG.evaluate(t)
        for t in simulate(plant, channel, encoder, decoder, LQG, T + 1)]

plt.plot(range(0, T + 1), LQG_trajectory)
plt.show()
