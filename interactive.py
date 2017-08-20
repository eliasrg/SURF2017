import sys, os
sys.path.insert(0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'code')))

from itertools import islice
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.integrate import quad

from simulation import Simulation, Parameters
from measurements import Measurement
import separate.coding.source.lloyd_max as lm
from separate.coding.convolutional import ConvolutionalCode, Node, \
        NaiveMLDecoder, StackDecoder
from utilities import hamming_distance
from joint.coding import SpiralMap


n_runs = 1 << 0
T = 1 << 7

average_measurements = []
def simulate(plots=False):
    global params, sim, average_measurements
    for SNR in [2]: # SNR irrelevant for an IntegerChannel
        print("SNR = {}".format(SNR))
        params = Parameters(
                T = T,
                alpha = 1.5,
                W = 1, V = 0, # Lloyd-Max paper assumes no observation noise
                Q = 1, R = 1, F = 1)

        # params.setRates(KC = 2, KS = 1)
        # params.setAnalog(SNR)
        # params.setScheme('joint')
        # print("SDR0 = {}".format(params.SDR0))

        # params.setDigital(quantizer_bits = 1)
        # params.setScheme('lloyd-max')

        # params.setDigital(quantizer_bits = 1, p = 0.1)
        # params.setBlocklength(3)
        # params.setScheme('noisy-lloyd-max')

        params.setRates(KC = 2, KS = 1)
        params.setAnalog(SNR)
        params.quantizer_bits = 1
        params.setBlocklength(2)
        params.setScheme('separate')

        measurements = []
        for i in range(n_runs):
            sim = Simulation(params)
            measurement = Measurement(params)
            if plots:
                tracker = sim.encoder.get_tracker()
                plot_lloyd_max(tracker.distr,
                        tracker.lm_encoder,
                        tracker.lm_decoder, x_hit=sim.plant.x)
            try:
                for t in sim.simulate(T):
                    print("Run {:d}, t = {:d}".format(i, t))
                    measurement.record(sim)
                    if plots:
                        tracker = sim.encoder.get_tracker()
                        plot_lloyd_max_tracker(tracker.distr,
                                tracker.lm_encoder,
                                tracker.lm_decoder,
                                tracker, x_hit=sim.plant.x)
            except KeyboardInterrupt:
                print("Keyboard interrupt!")
            measurements.append(measurement)

        average_measurements.append(Measurement.average(measurements))
        print("  Average power over channel: {:.4f}".format(
            sim.channel.average_power()))

    globals().update(params.all()) # Bring parameters into scope

def plot():
    figure = plt.figure()

    for average_measurement in average_measurements:
        plt.figure(figure.number)
        average_measurement.plot_setup()
        average_measurement.plot_LQG()


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
    plt.plot(x, y, 'lightblue')

    plt.axis('square')
    plt.axis([-22, 22, -22, 22])

def plot_spiral_decode():
    spiral_map = SpiralMap(2, 3)
    fig = plt.figure()
    plt.title("Closest point on Archimedean bi-spiral")
    plot_spiral(spiral_map)

    while True:
        # Retrieve a point that the user clicks
        points = []
        while not points:
            points = plt.ginput(1)
        received = points[0]

        s = spiral_map.decode(received)
        decoded = spiral_map.encode(s)

        plt.scatter([received[0]], [received[1]], color='tomato')
        plt.plot([received[0], decoded[0]], [received[1], decoded[1]],
                color='tomato')
        plt.scatter([decoded[0]], [decoded[1]], color='tomato')

        fig.canvas.draw()


def show(delay=0):
    if delay != 0:
        from time import sleep
        sleep(delay)
    plt.show(block=False)
