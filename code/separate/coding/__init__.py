from distributions import gaussian, Custom
from . import lloyd_max

import numpy as np
from scipy.integrate import quad, trapz
from scipy.signal import convolve
import scipy.stats as stats

# Integral subdivision limit
LIMIT = 10


class Encoder:
    def __init__(self, sim, mutual_state):
        self.sim = sim
        self.mutual_state = mutual_state

    def encode(self, *msg):
        # Just encode it with Lloyd-Max
        # (pass it to channel encoder or digital channel)
        return (self.mutual_state.lm_encoder.encode(*msg),)


class Decoder:
    def __init__(self, sim, mutual_state):
        self.sim = sim
        self.mutual_state = mutual_state

    def decode(self, *msg):
        # Decode with the Lloyd-Max decoder
        # (receive from channel encoder or digital channel)
        x_est = self.mutual_state.lm_decoder.decode(*msg)

        # Update the distribution tracker
        assert(len(msg) == 1) # One integer
        i = msg[0]
        self.mutual_state.update(i)

        return (x_est,)


class MutualState:
    """State known to both the encoder and the decoder (would be computed at
    both ends in practice, but here it is just computed once to save time)."""
    def __init__(self, sim, n_levels):
        self.sim = sim
        self.n_levels = n_levels

        self.distr = sim.plant.x1_distr
        self.lm_encoder, self.lm_decoder = \
            lloyd_max.generate(n_levels, self.distr)

        # DEBUG
        self.distrs = []

    def update(self, i, debug_globals=dict()):
        print("i = {}".format(i))
        # Retrieve the interval and reproduction value
        lo, hi = self.lm_decoder.get_interval(i)
        x_est = self.lm_decoder.decode(i)
        # (avoid infinite intervals)
        if i == 0: lo = hi - 3 * self.distr.std()
        if i == self.n_levels - 1: hi = lo + 3 * self.distr.std()

        # Compute parameters
        alpha = self.sim.params.alpha
        gamma, _ = quad(self.distr.pdf, lo, hi)
        mean, _ = quad(lambda x: x * self.distr.pdf(x) / gamma, lo, hi)
        print("mean - x_est = {}".format(mean - x_est))
        variance, _ = quad(lambda x: (x - x_est)**2 * self.distr.pdf(x) / gamma,
                lo, hi)
        std = np.sqrt(variance)
        next_std = alpha**2 * std

        # Discretize
        N_SAMPLES = 1000
        next_lo = -3 * next_std
        next_hi =  3 * next_std
        x = np.linspace(next_lo, next_hi, num=N_SAMPLES)
        fx = np.where((lo <= x / alpha + x_est) * (x / alpha + x_est <= hi),
                self.distr.pdf(x / alpha + x_est) / (alpha * gamma),
                0)
        spacing = x[1] - x[0]

        # Noise
        w_distr = self.sim.plant.w_distr
        w_x = np.arange(-2 * w_distr.std(), 2 * w_distr.std(), spacing)
        w_fx = w_distr.pdf(w_x)

        # DEBUG
        self.x = x
        self.fx = fx
        self.w_x = w_x
        self.w_fx = w_fx

        # Convolve
        fx_with_noise = spacing * convolve(fx, w_fx,
                mode='same', method='auto')

        # Normalize to compensate for cut-off
        fx_with_noise /= trapz(fx_with_noise, x)

        print("New mean: {}".format(trapz(x * fx_with_noise, x)))


        # Interpolate the new PDF and construct the new distribution
        self.distr = Custom(x, fx_with_noise)
        self.distrs.append(self.distr) # DEBUG

        # DEBUG: For inspecting the local variables interactively
        debug_globals.update(locals())

        # Generate the next Lloyd-Max quantizer
        self.lm_encoder, self.lm_decoder = lloyd_max.generate(
                self.n_levels, self.distr)
