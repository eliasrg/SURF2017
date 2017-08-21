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
from plotting import plot_lloyd_max, plot_lloyd_max_tracker, \
        plot_spiral, plot_spiral_decode

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


def show(delay=0):
    if delay != 0:
        from time import sleep
        sleep(delay)
    plt.show(block=False)
