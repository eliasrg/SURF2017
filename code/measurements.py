from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
import pickle

class Measurement:
    def __init__(self, params):
        self.params = params
        self.x = []
        self.LQG = []

    def record(self, sim):
        self.x.append(sim.plant.x)
        self.LQG.append(sim.LQG.evaluate(sim.t))

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
        assert all(m.params == measurements[0].params for m in measurements)
        new = Measurement(measurements[0].params)

        def average_sequence(sequences):
            slices = list(zip(*sequences))
            return np.array(list(map(np.mean, slices)))

        new.x = average_sequence(m.x for m in measurements)
        new.LQG = average_sequence(m.LQG for m in measurements)

        return new


    def plot(self):
        self.plot_setup()
        self.plot_LQG()

    def plot_setup(self):
        plt.xlabel("Time [steps]")

        plt.ylim(-10, 50)
        plt.grid()

    def plot_x(self):
        plt.plot(list(range(len(self.x))), self.x)
        plt.ylabel("Plant state [dB]")

    def plot_LQG(self):
        params = self.params

        plt.plot(list(range(len(self.LQG))), 10 * np.log10(self.LQG))
        plt.ylabel("Average LQG [dB]")

        # Lower bound
        if params.analog:
            plt.plot((1, len(self.LQG)),
                    10 * np.log10(params.LQR_inf_lower_bound()) * np.ones(2),
                    'r--')

        # Upper bound
        if params.analog and hasattr(params, 'SDR0'):
            plt.plot((1, len(self.LQG)),
                    10 * np.log10(params.LQR_inf_upper_bound()) * np.ones(2),
                    'g--')
