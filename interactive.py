# Copyright (c) 2017 Elias Riedel Gårding
# Licensed under the MIT License

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
        plot_lloyd_max_hikmet, plot_lloyd_max_tracker_hikmet, \
        plot_spiral, plot_spiral_decode

import separate.coding.source.lloyd_max as lm
from separate.coding.convolutional import ConvolutionalCode, Node, \
        NaiveMLDecoder, StackDecoder
import separate.coding.PAM as PAM
from utilities import *
from joint.coding import SpiralMap


n_runs = 1 << 0
T = 1 << 7


SNR = 2
params = Parameters(
        T = T,
        alpha = 1.5,
        W = 1, V = 0, # Lloyd-Max paper assumes no observation noise
        Q = 1, R = 0, F = 1)

# params.setRates(KC = 2, KS = 1)
# params.setAnalog(SNR)
# params.setScheme('joint')
# print("SDR0 = {}".format(params.SDR0))

# params.setDigital(quantizer_bits = 1)
# params.setScheme('lloyd-max')

# params.setDigital(quantizer_bits = 1, p = 0.1)
# params.setBlocklength(3)
# params.setScheme('noisy-lloyd-max')
# params.set_random_code()

params.setRates(KC = 2, KS = 1)
params.setAnalog(SNR)
params.quantizer_bits = 1
params.setBlocklength(2)
params.set_PAM()
params.setScheme('separate')
params.set_random_code()


measurements = []
def simulate(params=params, get_noise_record=lambda: None, plots=False):
    global sim, measurements

    for i in range(n_runs):
        sim = Simulation(params, get_noise_record())
        measurement = Measurement(params)
        measurements.append(measurement)
        if plots:
            tracker = sim.encoder.get_tracker().clone()
            prev_distr = tracker.distr
            prev_lm_encoder = tracker.lm_encoder
            prev_lm_decoder = tracker.lm_decoder
        try:
            for t in sim.simulate(T):
                measurement.record(sim)
                if plots:
                    if t == 1:
                        if hasattr(prev_distr, 'is_hikmet'):
                            plot_lloyd_max_hikmet(prev_distr,
                                    prev_lm_encoder.boundaries,
                                    prev_lm_decoder.levels,
                                    x_hit=sim.plant.x)
                        else:
                            plot_lloyd_max(prev_distr,
                                    prev_lm_encoder,
                                    prev_lm_decoder,
                                    x_hit=sim.plant.x)
                    else:
                        if hasattr(prev_distr, 'is_hikmet'):
                            plot_lloyd_max_tracker_hikmet(prev_distr,
                                    prev_lm_encoder.boundaries,
                                    prev_lm_decoder.levels,
                                    tracker.d1, tracker.fw, x_hit=sim.plant.x)
                        else:
                            plot_lloyd_max_tracker(prev_distr,
                                    prev_lm_encoder,
                                    prev_lm_decoder,
                                    tracker, x_hit=sim.plant.x)
                    tracker = sim.encoder.get_tracker().clone()
                    prev_distr = tracker.distr
                    prev_lm_encoder = tracker.lm_encoder
                    prev_lm_decoder = tracker.lm_decoder
                print("Run {:d}, t = {:d} done".format(i, t))
        except KeyboardInterrupt:
            print("Keyboard interrupt!")

        print("  Average power over channel: {:.4f}".format(
            sim.channel.average_power()))

    globals().update(params.all()) # Bring parameters into scope

def plot(average=True):
    figure = plt.figure()

    for measurement in measurements if not average \
            else [Measurement.average(measurements)]:
        plt.figure(figure.number)
        measurement.plot_setup()
        measurement.plot_LQG()
        measurement.plot_bounds()


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


def plot_compare():
    jscc = Measurement.load('data/joint/alpha_1.001_SNR_2_KC_32-runs.p')
    separate1 = Measurement.load('data/separate/alpha_1.001_SNR_2_KC_2--1.p')
    separate2 = Measurement.load('data/separate/alpha_1.001_SNR_2_KC_2--2.p')
    plt.figure()

    jscc.plot_setup()

    # Plot in the right order so that the legend reads top-down
    separate1.plot_LQG("Separation, single run")
    separate2.plot_LQG("Separation, single run")
    jscc.plot_LQG("Spiral JSCC, 32-run average")
    jscc.plot_bounds(upper_label="Theoretical prediction (spiral JSCC)")
    plt.legend()
    plt.text(25, 5, jscc.params.text_description(),
            bbox={'facecolor': 'white', 'edgecolor': 'gray'})

def plot_compare_2():
    jscc_avg = Measurement.load('data/joint/alpha_1.5_SNR_2_KC_2_256-runs.p')
    jscc = Measurement.load('data/comparison/alpha_1.5_SNR_2_KC_2--1-joint.p')
    sep = Measurement.load('data/comparison/alpha_1.5_SNR_2_KC_2--1-separate.p')

    jscc.plot_setup()

    sep.plot_LQG("Tandem with (2-PAM)$^2$")
    sep.plot_correctly_decoded()
    jscc.plot_LQG("Spiral JSCC, same noise sequences")
    jscc_avg.plot_LQG("Spiral JSCC, 256-run average")
    jscc.plot_bounds(upper_label="Theoretical prediction (spiral JSCC)")

    plt.legend(loc=(.55, .48))
    plt.text(40, 1.6, jscc.params.text_description(),
            bbox={'facecolor': 'white', 'edgecolor': 'gray'})

def load_varying_SNR():
    files = {'noiseless': [1,2,3,4],
             5:  [1,2,3,4,5,6],
             7:  [1,2,3,4,5,6],
             8:  [1,2,3,4,5,6],
             10: [1,2,3,4,5,6],
             15: [1,2,3,4,5,6],
             20: [1,2,3,4],
             25: [1,2,3,4,5,6]}
    return {SNRdB: [Measurement.load(
            'data/separate/varying-SNR/{}--{}.p'.format(
                str(SNRdB) + ('dB' if SNRdB != 'noiseless' else ''), run))
            for run in runs]
        for SNRdB, runs in files.items()}

def load_convergent_varying_SNR(data=None):
    good = {'noiseless': [1,2,3],
            5:  [1,5],
            7:  [],
            8:  [1,5,6],
            10: [1,2,3,6],
            15: [2,3,5,6],
            20: [1,2,3],
            25: [3,4,5]}

    if data is None:
        data = load_varying_SNR()
    return {SNRdB: [runs[i - 1] for i in good[SNRdB]]
            for SNRdB, runs in data.items()}

def average_convergent_varying_SNR(data=None):
    if data is None:
        data = load_convergent_varying_SNR()
    return {SNRdB: np.mean([10 * np.log10(m.LQG[-1]) for m in runs])
            for SNRdB, runs in data.items()}

def print_average_convergent_varying_SNR():
    data = load_varying_SNR()
    convergent_data = load_convergent_varying_SNR(data)
    average_data = average_convergent_varying_SNR(convergent_data)

    for SNRdB, average in average_data.items():
        print("{}: {:6g} ({}/{} runs)".format(
            str(SNRdB) + (" dB" if SNRdB != 'noiseless' else ""),
            average,
            len(convergent_data[SNRdB]), len(data[SNRdB])))


def show(delay=0):
    if delay != 0:
        from time import sleep
        sleep(delay)
    plt.show(block=False)
