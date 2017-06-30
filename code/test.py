import numpy as np
import matplotlib.pyplot as plt

from simulation import Simulation, Parameters


params = Parameters(
        T = 100,
        alpha = 2,
        P1 = 1, # Same as W
        W = 1, V = 1, SNR = 5, SDR0 = 1e100,
        Q = 1, R = 1, F = 1)

globals().update(params.all()) # Bring parameters into scope


sim = Simulation(params)


LQG_trajectory = [sim.LQG.evaluate(t) for t in sim.simulate(T + 1)]

plt.plot(range(0, T + 1), LQG_trajectory)
plt.show()
