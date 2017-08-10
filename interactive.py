import sys, os
sys.path.insert(0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'code')))

from itertools import islice
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.integrate import quad

from simulation import Simulation, Parameters
import separate.coding.source.lloyd_max as lm
from separate.coding.convolutional import ConvolutionalCode, Node, \
        NaiveMLDecoder, StackDecoder
from utilities import hamming_distance
from joint.coding import SpiralMap


n_runs = 1 << 0
T = 1 << 5

LQG_average_trajectories = []
def simulate(plots=False):
    global params, sim, LQG_average_trajectories
    for SNR in [2]: # SNR irrelevant for an IntegerChannel
        print("SNR = {}".format(SNR))
        params = Parameters(
                T = T,
                alpha = 2,
                W = 1, V = 0, # Lloyd-Max paper assumes no observation noise
                Q = 1, R = 1, F = 1)
        params.setRates(KC = 1, KS = 1)
        # params.setAnalog(SNR)
        # params.setScheme('joint')
        # params.setDigital(quantizer_bits = 1)
        # params.setScheme('lloyd-max')
        params.setDigital(quantizer_bits = 1, p = 0.03)
        params.setBlocklength(1)
        params.setScheme('noisy-lloyd-max')

        LQG_trajectories = []
        for i in range(n_runs):
            sim = Simulation(params)
            # LQG_trajectory = tuple(sim.LQG.evaluate(t)
            #                        for t in sim.simulate(T))
            LQG_trajectory = []
            if plots:
                tracker = (sim.encoder.tracker if params.scheme == 'lloyd-max'
                        else sim.encoder.source_encoder.tracker)
                plot_lloyd_max(tracker.distr,
                        tracker.lm_encoder,
                        tracker.lm_decoder, x_hit=sim.plant.x)
            try:
                for t in sim.simulate(T):
                    print("Run {:d}, t = {:d}".format(i, t))
                    LQG_trajectory.append(sim.LQG.evaluate(t))
                    if plots:
                        tracker = (sim.encoder.tracker
                                if params.scheme == 'lloyd-max'
                                else sim.encoder.source_encoder.tracker)
                        plot_lloyd_max_tracker(tracker.distr,
                                tracker.lm_encoder,
                                tracker.lm_decoder,
                                tracker, x_hit=sim.plant.x)
            except KeyboardInterrupt:
                print("Keyboard interrupt!")
            LQG_trajectories.append(LQG_trajectory)

        LQG_slices = list(zip(*LQG_trajectories))
        LQG_average_trajectory = np.array(list(map(np.mean, LQG_slices)))
        LQG_average_trajectories.append(LQG_average_trajectory)
        print("  Average power over channel: {:.4f}".format(
            sim.channel.average_power()))

    globals().update(params.all()) # Bring parameters into scope

def plot():
    plot = plt.figure()

    for LQG_average_trajectory in LQG_average_trajectories:
        # Plot in dB
        plt.xlabel("Time [steps]")
        plt.ylabel("Average LQR [dB]")
        plt.figure(plot.number)
        plt.plot(list(islice(range(0, T + 1), len(LQG_average_trajectory))),
            10 * np.log10(LQG_average_trajectory))

    if params.analog:
        plt.plot((1, T),
            10 * np.log10(params.LQR_inf_lower_bound()) * np.ones(2), 'r--')

    plt.ylim(0, 50)
    plt.grid()


def plot_lloyd_max(distr, enc, dec, x_hit=None):
    plt.figure()
    plt.scatter(dec.levels, np.zeros(len(dec.levels)), color='red')
    # plt.scatter(enc.boundaries, np.zeros(len(enc.boundaries)),
    #         color='purple', s=3)
    for boundary in enc.boundaries:
        plt.plot([boundary, boundary], [-0.01, distr.pdf(boundary)], color='gray')
    # plt.scatter([distr.mean()], [distr.pdf(distr.mean())], color='green')
    if x_hit is not None: plt.scatter([x_hit], [0], marker='x')
    a = max(distr.a, -20)
    b = min(distr.b, 20)
    x = np.linspace(a, b, num=10000)
    plt.plot(x, distr.pdf(x))
    # plt.xlim(-20, 20)
    # plt.ylim(-0.05, 0.4)
    plt.axis('tight')

def plot_lloyd_max_tracker(distr, enc, dec, tracker, x_hit=None):
    plt.figure()
    plt.scatter(dec.levels, np.zeros(len(dec.levels)), color='red')
    # plt.scatter(enc.boundaries, np.zeros(len(enc.boundaries)),
    #         color='purple', s=3)
    for boundary in enc.boundaries:
        plt.plot([boundary, boundary], [-0.01, distr.pdf(boundary)], color='gray')
    # plt.scatter([distr.mean()], [distr.pdf(distr.mean())], color='green')
    if x_hit is not None: plt.scatter([x_hit], [0], marker='x')
    a = max(distr.a, -20)
    b = min(distr.b, 20)
    x = np.linspace(a, b, num=10000)
    plt.plot(x, distr.pdf(x))
    plt.plot(tracker.x, tracker.fx, color='orange')
    plt.plot(tracker.w_x, tracker.w_fx, color='purple')
    # plt.xlim(-20, 20)
    # plt.ylim(-0.05, 0.4)
    plt.axis('tight')

def generate_plot_lloyd_max(n_levels):
    distr = st.norm
    enc, dec = lm.generate(n_levels, distr)
    plot_lloyd_max(distr, enc, dec)

def test_update(i=4):
    global tracker
    from separate.coding.source import DistributionTracker
    tracker = DistributionTracker(sim, 10)
    plot_lloyd_max(tracker.distr, tracker.lm_encoder, tracker.lm_decoder)
    tracker.update(i, debug_globals=globals())
    plot_lloyd_max(tracker.distr, tracker.lm_encoder, tracker.lm_decoder)

def plot_spiral(spiral_map):
    s = np.linspace(0, 7, num=1000)

    # Positive s
    x, y = list(zip(*map(spiral_map.encode, s)))
    plt.plot(x, y, 'orange')

    # Negative s
    x, y = list(zip(*map(spiral_map.encode, -s)))
    plt.plot(x, y, 'blue')

    plt.axis('square')

def plot_spiral_decode():
    spiral_map = SpiralMap(2, 3)
    fig = plt.figure()
    plot_spiral(spiral_map)

    while True:
        received = plt.ginput(1)[0]
        plt.scatter([received[0]], [received[1]], color='black')

        s = spiral_map.decode(received)
        decoded = spiral_map.encode(s)
        plt.scatter([decoded[0]], [decoded[1]], color='purple')
        plt.plot([received[0], decoded[0]], [received[1], decoded[1]],
                color='purple')

        fig.canvas.draw()


def show(delay=0):
    if delay != 0:
        from time import sleep
        sleep(delay)
    plt.show(block=False)
