# Copyright (c) 2017 Elias Riedel Gårding
# Licensed under the MIT License

from scipy.stats import bernoulli

class Plant:
    """The system {x(t+1) = α x(t) + w(t) + u(t); y(t) = x(t) + v(t)}."""
    def __init__(self, alpha, x1_distr, w_distr, v_distr,
            w_sequence=None, v_sequence=None, x1=None):
        self.alpha = alpha
        self.x1_distr = x1_distr
        self.w_distr = w_distr
        self.v_distr = v_distr
        self.w_sequence = w_sequence
        self.v_sequence = v_sequence

        self.w = self.get_w()
        self.v = self.get_v()

        self.x = x1_distr.rvs() if x1 is None else x1
        self.y = self.x + self.v

    def step(self, u):
        self.w = self.get_w()
        self.v = self.get_v()

        self.x = self.alpha * self.x + self.w + u
        self.y = self.x + self.v

    def get_w(self):
        if self.w_sequence is None:
            return self.w_distr.rvs()
        else:
            return self.w_sequence.pop(0)

    def get_v(self):
        if self.v_sequence is None:
            return self.v_distr.rvs()
        else:
            return self.v_sequence.pop(0)


class RealChannel:
    """Transmits real numbers with additive noise."""
    def __init__(self, n_distr, n_sequence=None):
        self.n_distr = n_distr
        self.n_sequence = n_sequence
        self.total_power = 0
        self.uses = 0

    def transmit(self, msg):
        """msg is a sequence of real numbers"""
        for a in msg:
            self.total_power += a**2
        self.uses += 1

        self.last_noise = []
        def get_noise():
            n = self.n_distr.rvs() if self.n_sequence is None \
                    else self.n_sequence.pop(0)
            self.last_noise.append(n)
            return n

        return (a + get_noise() for a in msg)

    def average_power(self):
        return self.total_power / self.uses

    def clone(self):
        return RealChannel(self.n_distr)

class IntegerChannel:
    """Noiseless fixed-rate digital channel.
    Transmits integers in the range [0, 2^R - 1]."""
    def __init__(self, n_symbols):
        self.n_symbols = n_symbols # 2^R where R is the (fixed) rate
        self.last_noise = None

    def transmit(self, msg):
        """msg is an array/list of integers no longer than 2^R."""
        assert len(msg) <= self.n_symbols
        assert all(isinstance(x, int) for x in msg)
        return msg

    def average_power(self):
        return float('nan')

    def clone(self):
        return IntegerChannel(self.n_symbols)

class BinarySymmetricChannel:
    """Flips each bit with probability p."""
    def __init__(self, p):
        self.p = p
        self.noise_distr = bernoulli(p)

    def transmit(self, msg):
        """msg is an array/list of bits (integers that are 1 or 0)."""
        assert all(x in [0,1] for x in msg)

        self.last_noise = []
        def get_noise():
            n = self.noise_distr.rvs()
            self.last_noise.append(n)
            return n

        return (x ^ get_noise() for x in msg)

    def average_power(self):
        return float('nan')

    def clone(self):
        return BinarySymmetricChannel(self.p)


class LQGCost:
    """A cost function of the form
    J(T) = 1/T (F x(T+1)² + Σ{t = 1..t}(Q x(t)² + R u(t)²))."""
    def __init__(self, plant, Q, R, F):
        self.plant = plant
        self.Q = Q
        self.R = R
        self.F = F

        self.x_sum = 0.0
        self.u_sum = 0.0
        self.last_x = plant.x

    def step(self, u):
        """To be called immediately after plant.step()"""
        self.x_sum += self.Q * self.last_x**2
        self.u_sum += self.R * u**2
        self.last_x = self.plant.x

    def evaluate(self, t):
        """If called at time t (i.e. after t-1 steps), will return J(t-1).

    The parameter t must agree with the current simulation time."""
        if t <= 1:
            return float('nan')
        else:
            return (self.x_sum + self.u_sum + self.F * self.last_x**2) / (t - 1)
