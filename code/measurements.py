# Copyright (c) 2017 Elias Riedel Gårding
# Licensed under the MIT License

from itertools import islice
from types import SimpleNamespace
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle

class Measurement:
    def __init__(self, params):
        self.params = params
        self.x = []
        self.w = []
        self.v = []
        self.noise = []
        self.LQG = []

        if params.scheme in ['noisy_lloyd_max', 'separate']:
            # Quantization index translated into bits
            self.bits = []
            # Entire history believed by the decoder (at each step)
            self.decoded_bits_history = []
            self.correctly_decoded = []

    def record(self, sim):
        self.x.append(sim.plant.x)
        self.w.append(sim.plant.w)
        self.v.append(sim.plant.v)
        self.noise.append(sim.channel.last_noise)
        self.LQG.append(sim.LQG.evaluate(sim.t))
        self.channel_average_power = sim.channel.average_power()

        if hasattr(self, 'bits'):
            self.bits = sim.encoder.get_bits_history()
            self.decoded_bits_history.append(list(
                    sim.decoder.stack_decoder.first_nodes[-1].input_history()))
            self.correctly_decoded.append(
                    all((word == history_word).all()
                        for word, history_word in \
                                zip(self.bits, self.decoded_bits_history[-1])))
            print("Correctly decoded: {}".format(self.correctly_decoded[-1]))

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            measurement = pickle.load(f)
            assert isinstance(measurement, Measurement)
            return measurement

    @staticmethod
    def average(measurements):
        new = Measurement(measurements[0].params)

        def average_sequence(sequences):
            sequences = [np.array(sequence).flatten() for sequence in sequences]
            slices = list(zip(*sequences))
            return np.array(list(map(np.mean, slices)))

        new.x = average_sequence(m.x for m in measurements)
        new.w = average_sequence(m.w for m in measurements)
        new.v = average_sequence(m.v for m in measurements)
        new.noise = average_sequence(m.noise for m in measurements)
        new.LQG = average_sequence(m.LQG for m in measurements)

        return new

    def get_noise_record(self):
        noise = SimpleNamespace()

        noise.x1 = self.x[0]
        noise.w_sequence = self.w[:]
        noise.v_sequence = self.v[:]
        noise.n_sequence = list(np.array(self.noise).flatten())

        return noise


    def plot(self, label=None):
        self.plot_setup()
        self.plot_LQG(label=label)
        self.plot_bounds()
        if hasattr(self, 'correctly_decoded'):
            self.plot_correctly_decoded()
        plt.legend()

    def plot_setup(self, label="Time [steps]"):
        plt.xlabel(label)
        plt.grid()

    def plot_x(self):
        plt.plot(list(range(len(self.x))), self.x)
        plt.ylabel("Plant state")

    def plot_LQG(self, label=None, *args, **kwargs):
        plt.plot(list(range(len(self.LQG))), 10 * np.log10(self.LQG),
                label=label, *args, **kwargs)
        plt.ylabel(r"$\bar{J}_t$ [dB]")

    def plot_bounds(self, lower_label="Theoretical average lower bound",
            upper_label="Theoretical prediction",
            lower_args=['--'], lower_kwargs={},
            upper_args=['--'], upper_kwargs={}):
        params = self.params

        # Upper bound
        if params.analog and hasattr(params, 'SDR0'):
            plt.plot((1, len(self.LQG)),
                    10 * np.log10(params.LQR_inf_upper_bound()) * np.ones(2),
                    *upper_args, label=upper_label, **upper_kwargs)

        # Lower bound
        if params.analog:
            plt.plot((1, len(self.LQG)),
                    10 * np.log10(params.LQR_inf_lower_bound()) * np.ones(2),
                    *lower_args, label=lower_label, **lower_kwargs)

    def plot_correctly_decoded(self, y=0):
        RECTANGLE_HEIGHT = 0.8

        # Find intervals of consecutive Trues
        intervals = []
        start = None
        for (t, good) in enumerate(self.correctly_decoded, 1):
            if not start and not good:
                start = t
            elif start and (good or t == len(self.correctly_decoded)):
                intervals.append((start, t))
                start = None

        for i, (start, stop) in enumerate(intervals):
            print("({}, {})".format(start, stop))
            plt.gca().add_patch(
                patches.Rectangle(
                    (start, y - RECTANGLE_HEIGHT/2),
                    stop - start,
                    RECTANGLE_HEIGHT,
                    label="Decoding errors" if i == 0 else None,
                    color='purple'
                )
            )
