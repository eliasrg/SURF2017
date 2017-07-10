from distributions import gaussian
import separate.coding.lloyd_max as lloyd_max

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.signal import convolve
import scipy.stats as stats

# Integral subdivision limit
LIMIT = 10

class MutualState:
    """State known to both the observer and the controller."""
    def __init__(self, sim, n_levels):
        self.sim = sim
        self.n_levels = n_levels

        self.distr = gaussian(sim.params.W)
        self.encoder, self.decoder = lloyd_max.generate(n_levels, self.distr)

    def update(self, i, debug_globals=dict()):
        # Cut and discretize
        lo, hi = self.decoder.get_interval(i)
        # Note: Using CDF in this case would be slightly _less_ efficient
        gamma, _ = quad(self.distr.pdf, lo, hi, limit=LIMIT) # (below (11))
        N_SAMPLES = 50
        x = np.linspace(lo, hi, num=N_SAMPLES)
        fx = self.distr.pdf(x) / gamma # (11)

        # Predict without noise
        alpha = self.sim.params.alpha
        x_hat = self.decoder.decode(i)
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
        self.encoder, self.decoder = lloyd_max.generate(
                self.n_levels, self.distr)


class Observer:
    """The observer/transmitter described in the Lloyd-Max paper."""
    def __init__(self, sim):
        self.sim = sim

    def observe(self, t, y):
        pass # TODO


class Controller:
    """The controller/receiver described in the Lloyd-Max paper."""
    def __init__(self, sim):
        self.sim = sim

    def control(self, t, *msg):
        sim = self.sim

        pass # TODO
