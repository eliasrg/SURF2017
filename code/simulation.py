from distributions import onepoint, zero, gaussian
from system import Plant, Channel, LQGCost
from coding import TrivialEncoder, TrivialDecoder
from utilities import memoized

from types import SimpleNamespace

class Simulation:
    def __init__(self, params):
        self.params = params

        self.plant = Plant(params.alpha, gaussian(params.P0),
                gaussian(params.W), gaussian(params.V))
        self.channel = Channel(gaussian(1 / params.SNR))
        self.encoder = TrivialEncoder()
        self.decoder = TrivialDecoder()

        self.LQG = LQGCost(self.plant, params.Q, params.R, params.F)

        # Globally known data
        self.globals = SimpleNamespace()
        self.globals.u = dict()

    def simulate(self, T):
        t = 1
        yield t
        while t < T:
            x_est = self.decoder.decode(self, t,
                    *(self.channel.transmit(p)
                        for p in self.encoder.encode(self, t, self.plant.y)))
            u = -self.plant.alpha * x_est
            self.plant.step(u)
            self.LQG.step(u)
            self.globals.u[t] = u
            t += 1
            yield t


class Parameters:
    def __init__(self, T, alpha, P0, W, V, SNR, Q, R, F):
        self.T = T # Time horizon
        self.alpha = alpha # System coefficient
        assert(alpha > 1) # unstable

        # Variance (noise power) parameters
        self.P0 = P0 # V[x0]
        self.W = W # V[w_t]
        self.V = V # V[v_t]
        self.SNR = SNR # 1 / V[n_t]

        # LQG parameters
        self.Q = Q
        self.R = R
        self.F = F

    def all(self):
        names = ['T', 'alpha', 'P0', 'W', 'V', 'SNR', 'Q', 'R', 'F']
        return {name: self.__dict__[name] for name in names}

    # Statically known parameters computed recursively using memoization

    @memoized
    def Pt(self, t, t_obs):
        if (t, t_obs) == (1, 0):
            return self.P0
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
            return self.P0
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
        if t == self.T:
            return self.F
        elif 1 <= t < self.T:
            S = self.S(t+1)
            return self.alpha**2 * self.R * S / (S + self.R) + self.Q
        else:
            raise ValueError(("S({}) undefined because {} is out of the range "
                    + "[1, T = {}]").format(t, t, self.T))
