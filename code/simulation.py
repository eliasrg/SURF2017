from distributions import onepoint, zero, gaussian
from system import Plant, Channel, LQGCost
from coding import TrivialEncoder, TrivialDecoder, Encoder, Decoder
from utilities import memoized

from types import SimpleNamespace
import numpy as np

class Simulation:
    def __init__(self, params):
        self.params = params

        # Globally known data
        self.globals = SimpleNamespace()
        # u[t]: control signal at time t
        self.globals.u = dict()
        # x_est_r[t1, t2]: receiver estimate of x(t1) at time t2
        self.globals.x_est_r = dict()

        self.plant = Plant(params.alpha, gaussian(params.P1),
                gaussian(params.W), gaussian(params.V))
        self.channel = Channel(gaussian(1 / params.SNR))
        self.encoder = Encoder()
        self.decoder = Decoder(self)

        self.LQG = LQGCost(self.plant, params.Q, params.R, params.F)

    def simulate(self, T):
        t = 1
        while t <= T:
            u = self.decoder.decode(self, t,
                    *(self.channel.transmit(p)
                        for p in self.encoder.encode(self, t, self.plant.y)))
            self.globals.u[t] = u
            yield t
            self.plant.step(u)
            self.LQG.step(u)
            t += 1
        yield t


class Parameters:
    def __init__(self, T, alpha, P1, W, V, SNR, SDR0, Q, R, F, KC, KS):
        self.T = T # Time horizon
        self.alpha = alpha # System coefficient
        assert(alpha > 1) # unstable

        # Variance (noise power) parameters
        self.P1 = P1 # V[x_1]
        self.W = W # V[w_t]
        self.V = V # V[v_t]
        self.SNR = SNR # 1 / V[n_t]
        self.SDR0 = SDR0 # Channel code signal-distortion ratio, 1 / V[neff_t]

        # LQG parameters
        self.Q = Q
        self.R = R
        self.F = F

        # Channel usage parameters
        self.KC = KC
        self.KS = KS

        # Pre-evaluate L(t) in the right order to avoid blowing the stack
        for t in range(self.T, 0, -1):
            self.L(t)

    def all(self):
        names = ['T', 'alpha', 'P1', 'W', 'V', 'SNR', 'Q', 'R', 'F', 'KC', 'KS']
        return {name: self.__dict__[name] for name in names}

    # Statically known parameters computed recursively using memoization

    @memoized
    def Pt(self, t, t_obs):
        if (t, t_obs) == (1, 0):
            return self.P1
        elif t_obs == t:
            return self.Kt(t) * self.V
        elif t_obs == t-1:
            return self.alpha**2 * self.Pt(t-1, t-1) + self.W
        else:
            raise ValueError("({}, {}) not on the form (t, t) or (t+1, t)"
                             .format(t, t_obs))

    @memoized
    def Kt(self, t):
        P = self.Pt(t, t-1)
        return P / (P + self.V)

    @memoized
    def Pr(self, t, t_obs):
        if (t, t_obs) == (1, 0):
            return self.P1
        elif t_obs == t:
            return ((self.Pr(t, t-1) + self.SDR0 * self.Pt(t, t))
                    / (1 + self.SDR0))
        elif t_obs == t-1:
            return self.alpha**2 * self.Pr(t-1, t-1) + self.W
        else:
            raise ValueError("({}, {}) not on the form (t, t) or (t+1, t)"
                             .format(t, t_obs))

    @memoized
    def L(self, t):
        S = self.S(t+1)
        return self.alpha * S / (S + self.R)

    @memoized
    def S(self, t):
        if t == self.T + 1:
            return self.F
        elif 1 <= t <= self.T:
            S = self.S(t+1)
            return self.alpha**2 * self.R * S / (S + self.R) + self.Q
        else:
            raise ValueError(("S({}) undefined because {} is out of the range "
                    + "[1, T + 1 = {}]").format(t, t, self.T + 1))

    # Infinite horizon values
    @memoized
    def S_inf(self):
        return next(filter(lambda S: S > 0,
                np.polynomial.polynomial.Polynomial(list(reversed([
                    1,
                    -(self.Q + (self.alpha**2 - 1) * self.R),
                    -self.Q * self.R]))).roots()))

    @memoized
    def Pt_inf(self):
        return next(filter(lambda S: S > 0,
                np.polynomial.polynomial.Polynomial(list(reversed([
                    1,
                    -((self.alpha**2 - 1) * self.V + self.W),
                    -self.V * self.W]))).roots()))

    @memoized
    def Pt_inf_bar(self):
        Pt = self.Pt_inf()
        return Pt * self.V / (Pt + self.V)

    @memoized
    def LQR_inf_classical(self):
        Pt = self.Pt_inf()
        Pt_bar = self.Pt_inf_bar()
        return self.Q * Pt_bar + self.S_inf() * (Pt - Pt_bar)

    @memoized
    def LQR_inf_upper_bound(self):
        Pt = self.Pt_inf()
        Pt_bar = self.Pt_inf_bar()
        return (self.LQR_inf_classical()
                + (self.Q + (self.alpha**2 - 1) * self.S_inf())
                    / (1 + self.SDR0 - self.alpha**2)
                  * (Pt - Pt_bar))

    @memoized
    def LQR_inf_lower_bound(self):
        Pt = self.Pt_inf()
        Pt_bar = self.Pt_inf_bar()
        return (self.LQR_inf_classical()
                + (self.Q + (self.alpha**2 - 1) * self.S_inf())
                    / (1 + self.SDR_OPTA() - self.alpha**2)
                  * (Pt - Pt_bar))

    @memoized
    def SDR_OPTA(self):
        return (1 + self.SNR)**(self.KC / self.KS) - 1
