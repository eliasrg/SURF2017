# Copyright (c) 2017 Elias Riedel Gårding
# Licensed under the MIT License

import sys, os
sys.path.insert(0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'code')))

from itertools import islice, count
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.integrate import quad
from collections import defaultdict

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


SNR = 10**(20 / 10)
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

params.setDigital(quantizer_bits = 1, p = 0.001)
params.setBlocklength(2)
params.setScheme('noisy-lloyd-max')
params.set_random_code()

params.setRates(KC = 2, KS = 1)
params.setAnalog(SNR)
params.quantizer_bits = 1
params.setBlocklength(2)
params.set_PAM()
params.setScheme('separate')
params.set_random_code()


def generate_filename(SNR_dB, alpha, i, quantizer_bits=1, code_blocklength=1):
    if SNR_dB == 'noiseless':
        filename_pattern = \
            'data/separate/varying-SNR/alpha{}/noiseless/noiseless--{{}}.p' \
            .format(alpha)
    else:
        filename_pattern = 'data/separate/varying-SNR/alpha{}/{}:{}/{}dB--{{}}.p' \
                .format(alpha, quantizer_bits, code_blocklength, SNR_dB)

    return filename_pattern.format(i)

def load_measurements(SNR_dB, alpha=1.2, quantizer_bits=1, code_blocklength=2):
    results = []
    for i in count(1):
        filename = generate_filename(SNR_dB, alpha, i, quantizer_bits,
                code_blocklength)
        if os.path.isfile(filename):
            results.append(Measurement.load(filename))
        else:
            break

    return results

def simulate_and_record(params):
    # Take measurement
    bad = False
    if SNR_dB != 'noiseless':
        params.set_random_code() # Use different codes each time
    try:
        simulate(params)
    except (ValueError, TypeError):
        input("Bad! :( ")
        bad = True

    # Generate filename
    for i in count(1):
        filename = 'bad-' if bad else ''
        filename += generate_filename(SNR_dB, params.alpha, i,
                params.quantizer_bits,
                params.code_blocklength)
        if not os.path.isfile(filename):
            break

    print("Saving to {}".format(filename))
    measurements[-1].save(filename)



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

def plot_compare_3():
    import matplotlib
    matplotlib.rcParams.update({'font.size': 20, 'lines.linewidth': 3})

    jscc = Measurement.load('data/comparison/alpha_1.2_SNR_4.5dB_KC_2--1-joint.p')
    sep = Measurement.load('data/comparison/alpha_1.2_SNR_4.5dB_KC_2--1-separate.p')

    jscc.plot_setup(label="t")

    sep.plot_LQG("2-PAM", ':')
    sep.plot_correctly_decoded(y=-15)
    jscc.plot_LQG("Spiral", '-')
    jscc.plot_bounds(upper_label="Spiral: analytic", lower_label="OPTA",
            upper_args=['-.'], lower_args=['--'])

    plt.legend(loc=(.4, .1))

def plot_varying_SNR(alpha, multi=False, log_outside=True):
    plt.figure()

    def closest(n):
        return min(256, 512, 1024, 1536, 2048, key=lambda x: abs(x - n))

    # SNR_dBs = [9, 10, 10.5, 11, 11.5, 12, 13, 13.5, 14, 15, 17, 20, 23, 25]
    # SNR_dBs = [7.5, 8, 8.5, 9, 10, 10.25, 10.5, 10.75, 11, 11.25, 11.5, 11.75, 12, 13,
    #         14, 15, 16, 'noiseless']
    if not multi:
        # SNR_dBs = sorted([
        #             # 10, 10.25, 10.5, 10.75,
        #             11, 11.25, 11.5, 11.75,
        #             12, 12.25, 12.5, 12.75,
        #             13,        13.5, 13.75,
        #             14, 15, 16
        #             ])
        SNR_dBs = sorted(
                # [1, 1.5, 2, 2.5, 3, 3.5] +
                [4, 4.25, 4.5, 5, 5.5, 6, 6.5, 7, 15, 25])
        ms = {SNR_dB: load_measurements(SNR_dB, alpha) for SNR_dB in SNR_dBs}
    else:
        # SNR_dBs = sorted([30, 25, 24.5, 24, 23.5, 23, 22.5, 22, 20, 17.5, 15, 12.5, 10])
        # SNR_dBs = sorted([8, 25, 24.5, 24, 23.5, 23, 22.75, 22.5, 22])
        SNR_dBs = sorted([7, 8, 8.5, 9, 9.5, 10, 11, 12, 15, 25])
        ms = {SNR_dB: load_measurements(SNR_dB, alpha, 2, 4)
                for SNR_dB in SNR_dBs}

    if log_outside:
        LQGlog10s = [10 * np.log10(np.mean([m.LQG[-1] for m in ms[SNR_dB]]))
                for SNR_dB in SNR_dBs]
    else:
        LQGlog10s = [np.mean([10 * np.log10(m.LQG[-1]) for m in ms[SNR_dB]])
                for SNR_dB in SNR_dBs]

    plt.grid()
    SNR_dBs += [None]
    LQGlog10s += [None]
    plt.scatter(SNR_dBs[:-1], LQGlog10s[:-1]) # (without noiseless)
    plt.xlabel("SNR [dB]")
    plt.ylabel("Average final average cost [dB]")

    del SNR_dBs[-1], LQGlog10s[-1]

    for SNR_dB, LQGlog10 in zip(SNR_dBs, LQGlog10s):
        print("{:5}dB: {} ({} runs)".format(SNR_dB, LQGlog10,
            len(ms[SNR_dB]) ))#,
            # closest(len(ms[SNR_dB]))))


def show(delay=0):
    if delay != 0:
        from time import sleep
        sleep(delay)
    plt.show(block=False)



def take_data():
    global SNR_dB
    for alpha in [1.2]: #, 1.2]:
        for SNR_dB in [
                # 11, 12
                # 8.5, 9.5
                15, 25
                ]:
            SNR = 10**(SNR_dB / 10)
            params = Parameters(
                    T = T,
                    alpha = alpha,
                    W = 1, V = 0, # Lloyd-Max paper assumes no observation noise
                    Q = 1, R = 0, F = 1)

            params.setRates(KC = 2, KS = 1)
            params.setAnalog(SNR)
            params.quantizer_bits = 1
            params.setBlocklength(2)
            params.set_PAM()
            params.setScheme('separate')
            params.set_random_code()

            N = 256
            n = defaultdict(lambda: N, {3: 2 * 256 - 280})

            for _ in range(n[SNR_dB]):
                try:
                    simulate_and_record(params)
                except KeyboardInterrupt:
                    break
                except (ValueError, TypeError):
                    pass


            del measurements[:]
