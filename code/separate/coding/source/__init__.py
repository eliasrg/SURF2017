# Copyright (c) 2017 Elias Riedel Gårding
# Licensed under the MIT License

from distributions import gaussian, Custom
from . import lloyd_max

import numpy as np
from scipy.integrate import quad, trapz
from scipy.signal import convolve
import scipy.stats as stats

# Integral subdivision limit
LIMIT = 10


class Encoder:
    def __init__(self, sim, tracker):
        self.sim = sim
        self.tracker = tracker

    def encode(self, *msg):
        # Just encode it with Lloyd-Max
        # (pass it to channel encoder or digital channel)
        i = self.tracker.lm_encoder.encode(*msg)
        self.tracker.update(i)
        return (i,)

    def get_tracker(self):
        return self.tracker


class Decoder:
    def __init__(self, sim, tracker):
        self.sim = sim
        self.tracker = tracker

    def clone(self):
        return self.__class__(self.sim, self.tracker.clone())

    def decode(self, *msg):
        # Decode with the Lloyd-Max decoder
        # (receive from channel encoder or digital channel)
        x_est = self.tracker.lm_decoder.decode(*msg)

        # Update the distribution tracker
        assert len(msg) == 1 # One integer
        i = msg[0]
        self.tracker.update(i)

        return (x_est,)


class DistributionTracker:
    """Keeps track of the distribution of the plant's state."""
    def __init__(self, sim, n_levels, distr=None,
            lm_encoder=None, lm_decoder=None):
        self.sim = sim
        self.n_levels = n_levels

        if distr is None:
            assert lm_encoder is None and lm_decoder is None
            self.distr = sim.plant.x1_distr
            self.lm_encoder, self.lm_decoder = \
                lloyd_max.generate(n_levels, self.distr)
        else:
            assert lm_encoder is not None and lm_decoder is not None
            self.distr = distr
            self.lm_encoder = lm_encoder
            self.lm_decoder = lm_decoder

        # DEBUG
        self.distrs = []

    def clone(self):
        new = self.__class__(self.sim, self.n_levels, self.distr,
                self.lm_encoder, self.lm_decoder)

        # DEBUG
        new.distrs = self.distrs[:]
        if hasattr(self, 'x'): new.x = self.x
        if hasattr(self, 'fx'): new.fx = self.fx
        if hasattr(self, 'w_x'): new.w_x = self.w_x
        if hasattr(self, 'w_fx'): new.w_fx = self.w_fx

        return new

    def update(self, i, debug_globals=dict()):
        # Retrieve the interval and reproduction value
        lo, hi = self.lm_decoder.get_interval(i)
        x_est = self.lm_decoder.decode(i)
        # (avoid infinite intervals)
        if i == 0: lo = hi - 3 * self.distr.std()
        if i == self.n_levels - 1: hi = lo + 3 * self.distr.std()

        # Compute parameters
        alpha = self.sim.params.alpha
        L = self.sim.params.L(self.sim.t)
        gamma, _ = quad(self.distr.pdf, lo, hi)
        mean, _ = quad(lambda x: x * self.distr.pdf(x) / gamma, lo, hi)
        variance, _ = quad(lambda x: (x - x_est)**2 * self.distr.pdf(x) / gamma,
                lo, hi)
        std = np.sqrt(variance)
        next_std = alpha * std

        # Discretize
        N_SAMPLES = 1000
        next_lo = -3 * next_std - 2
        next_hi =  3 * next_std + 2
        x = np.linspace(next_lo, next_hi, num=N_SAMPLES)
        fx = np.where((lo <= (x + L * x_est) / alpha) * ((x + L * x_est) / alpha <= hi),
                self.distr.pdf((x + L * x_est) / alpha) / (alpha * gamma),
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
                mode='same')

        # Normalize to compensate for cut-off
        fx_with_noise /= trapz(fx_with_noise, x)


        # Interpolate the new PDF and construct the new distribution
        self.distr = Custom(x, fx_with_noise)
        self.distrs.append(self.distr) # DEBUG

        # DEBUG: For inspecting the local variables interactively
        debug_globals.update(locals())

        # Generate the next Lloyd-Max quantizer
        self.lm_encoder, self.lm_decoder = lloyd_max.generate(
                self.n_levels, self.distr)
