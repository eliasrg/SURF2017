from distributions import gaussian
from . import lloyd_max

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
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

        return x_est


class MutualState:
    """State known to both the encoder and the decoder (would be computed at
    both ends in practice, but here it is just computed once to save time)."""
    def __init__(self, sim, n_levels):
        self.sim = sim
        self.n_levels = n_levels

        self.distr = gaussian(sim.params.W)
        self.lm_encoder, self.lm_decoder = \
            lloyd_max.generate(n_levels, self.distr)

    def update(self, i, debug_globals=dict()):
        # Cut and discretize
        lo, hi = self.lm_decoder.get_interval(i)
        # Note: Using CDF in this case would be slightly _less_ efficient
        gamma, _ = quad(self.distr.pdf, lo, hi, limit=LIMIT) # (below (11))
        N_SAMPLES = 50
        x = np.linspace(lo, hi, num=N_SAMPLES)
        fx = self.distr.pdf(x) / gamma # (11)

        # Predict without noise
        alpha = self.sim.params.alpha
        x_hat = self.lm_decoder.decode(i)
        next_x_without_noise = alpha * (x - x_hat) # also linearly spaced!
        spacing = next_x_without_noise[1] - next_x_without_noise[0]

        # Introduce noise
        w_distr = self.sim.plant.w_distr
        noise_x = np.arange(-2 * w_distr.std(), 2 * w_distr.std(), spacing)
        noise_fx = w_distr.pdf(noise_x)

        # Extra x space of 2σ_w on both sides
        next_lo = next_x_without_noise[0] - 2 * w_distr.std()
        next_hi = next_x_without_noise[-1] + 2 * w_distr.std()
        extra_x_below = np.array(sorted(np.arange(
                next_x_without_noise[0] - spacing, next_lo, -spacing)))
        extra_x_above = np.arange(
                next_x_without_noise[-1] + spacing, next_hi, spacing)
        next_x = np.concatenate((
            extra_x_below, next_x_without_noise, extra_x_above))
        next_fx_without_noise = np.concatenate((
            np.zeros(len(extra_x_below)),
            fx / alpha,
            np.zeros(len(extra_x_above))))

        # Convolve (automatically chooses direct or FFT method)
        next_fx = spacing * convolve(next_fx_without_noise, noise_fx,
                mode='same', method='auto')

        # Interpolate the new PDF
        pdf = interp1d(next_x, next_fx,
                kind='linear', bounds_error=False, fill_value=0)

        # Construct the new distribution
        self.distr = stats.rv_continuous(a=next_lo, b=next_hi)
        self.distr._pdf = pdf

        # DEBUG: For inspecting the local variables interactively
        debug_globals.update(locals())

        # Generate the next Lloyd-Max quantizer
        self.lm_encoder, self.lm_decoder = lloyd_max.generate(
                self.n_levels, self.distr)
