import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.integrate import quad

from simulation import Simulation, Parameters
import separate.coding.lloyd_max as lm


n_runs = 1 << 6
T = 1 << 5

LQG_average_trajectories = []
def simulate():
    global params, sim, LQG_average_trajectories
    for SNR in [2, 4]:
        print("SNR = {}".format(SNR))
        params = Parameters(
                T = T,
                alpha = 2,
                W = 1, V = 1,
                Q = 1, R = 1, F = 1)
        params.setRates(KC = 1, KS = 1)
        # params.setAnalog(SNR)
        # params.setScheme('joint')
        params.setDigital(n_codewords = 2)
        params.setScheme('lloyd-max')

        LQG_trajectories = []
        for i in range(n_runs):
            sim = Simulation(params)
            LQG_trajectory = tuple(sim.LQG.evaluate(t)
                                   for t in sim.simulate(T))
            LQG_trajectories.append(LQG_trajectory)

        LQG_slices = list(zip(*LQG_trajectories))
        LQG_average_trajectory = np.array(list(map(np.mean, LQG_slices)))
        LQG_average_trajectories.append(LQG_average_trajectory)
        print("  Average power over channel: {:.4f}".format(
            sim.channel.average_power()))

    globals().update(params.all()) # Bring parameters into scope

def plot():
    plt.figure()

    for LQG_average_trajectory in LQG_average_trajectories:
        # Plot in dB
        plt.xlabel("Time [steps]")
        plt.ylabel("Average LQR [dB]")
        plt.plot(range(0, T + 1), 10 * np.log10(LQG_average_trajectory))

    plt.plot((1, T),
            10 * np.log10(params.LQR_inf_lower_bound()) * np.ones(2), 'r--')

    plt.ylim(0, 50)
    plt.grid()
    plt.show(block=False)


def plot_lloyd_max(distr, enc, dec):
    plt.figure()
    plt.scatter(dec.levels, np.zeros(len(dec.levels)), color='red')
    plt.scatter(enc.boundaries, np.zeros(len(enc.boundaries)),
            color='purple', s=3)
    x = np.linspace(-4, 4, num=1000)
    plt.plot(x, distr.pdf(x))
    plt.show(block=False)

def generate_plot_lloyd_max(n_levels):
    distr = st.norm
    enc, dec = lm.generate(n_levels, distr)
    plot_lloyd_max(distr, enc, dec)

def test_update(i=4):
    global ms, ctrl
    import separate.coding as cdng
    ms = cdng.MutualState(sim, 10)
    plot_lloyd_max(ms.distr, ms.lm_encoder, ms.lm_decoder)
    ms.update(i, debug_globals=globals())
    plot_lloyd_max(ms.distr, ms.lm_encoder, ms.lm_decoder)
