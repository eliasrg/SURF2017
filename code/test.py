import numpy as np
import matplotlib.pyplot as plt

from simulation import Simulation, Parameters


n_runs = 100
T = 100

LQG_average_trajectories = []
for SNR in [2, 4]:
    params = Parameters(
            T = T,
            alpha = 2,
            P1 = 1, # Same as W
            W = 1, V = 1, SNR = SNR, SDR0 = SNR,
            Q = 1, R = 1, F = 1)

    # globals().update(params.all()) # Bring parameters into scope

    LQG_trajectories = []
    for i in range(n_runs):
        sim = Simulation(params)
        LQG_trajectory = tuple(sim.LQG.evaluate(t)
                               for t in sim.simulate(T + 1))
        LQG_trajectories.append(LQG_trajectory)

    LQG_slices = list(zip(*LQG_trajectories))
    LQG_average_trajectory = np.array(list(map(np.mean, LQG_slices)))
    LQG_average_trajectories.append(LQG_average_trajectory)

for LQG_average_trajectory in LQG_average_trajectories:
    # Plot in dB
    plt.xlabel("Time [steps]")
    plt.ylabel("Average LQR [dB]")
    plt.plot(range(0, T + 1), 10 * np.log10(LQG_average_trajectory))
    # plt.plot(range(0, T + 1), LQG_average_trajectory)

plt.ylim(0, 50)
plt.grid()
plt.show()
