from distributions import gaussian
from system import Plant, RealChannel, IntegerChannel, LQGCost
import joint.control
import separate.control
import separate.coding.source
import trivial.coding
from utilities import memoized

from types import SimpleNamespace
import numpy as np
import inspect

class Simulation:
    def __init__(self, params):
        self.params = params

        self.plant = Plant(params.alpha, gaussian(params.W),
                gaussian(params.W), gaussian(params.V))
        self.LQG = LQGCost(self.plant, params.Q, params.R, params.F)

        if params.analog:
            # Globally known data
            self.globals = SimpleNamespace()
            # u[t]: control signal at time t
            self.globals.u = dict()
            # x_est_r[t1, t2]: receiver estimate of x(t1) at time t2
            self.globals.x_est_r = dict()

            self.channel = RealChannel(gaussian(1 / params.SNR))
        else:
            self.channel = IntegerChannel(params.n_codewords)


        if params.scheme == 'joint':
            self.observer = joint.control.Observer(self)
            self.controller = joint.control.Controller(self)
            if (params.KC, params.KS) == (1, 1):
                self.encoder = trivial.coding.Encoder(self)
                self.decoder = trivial.coding.Decoder(self)
            elif (params.KC, params.KS) == (2, 1):
                pass # TODO Spiral encoder/decoder

        elif params.scheme == 'separate':
            pass # TODO Tree codes

        elif params.scheme == 'lloyd-max':
            self.observer = separate.control.Observer(self)
            self.controller = separate.control.Controller(self)

            self.tracker = separate.coding.source.DistributionTracker(
                    self, params.n_codewords)
            self.encoder = \
                    separate.coding.source.Encoder(self, self.tracker)
            self.decoder = \
                    separate.coding.source.Decoder(self, self.tracker)

    def simulate(self, T):
        t = 1
        while t <= T:
            # The observer observes the plant and generates a message
            msg = self.observer.observe(t, self.plant.y)
            # The encoder encodes the message
            code = self.encoder.encode(*msg)
            # The encoded message is sent over the channel
            code_recv = self.channel.transmit(code)
            # The decoder decodes the encoded message
            msg_recv = self.decoder.decode(*code_recv)
            # The controller receives the message and generates a control signal
            u = self.controller.control(t, *msg_recv)
            if self.params.analog:
                self.globals.u[t] = u

            yield t

            self.plant.step(u)
            self.LQG.step(u)

            t += 1
        yield t


class Parameters:
    def __init__(self, T, alpha, W, V, Q, R, F):
        self.T = T # Time horizon
        self.alpha = alpha # System coefficient
        assert alpha > 1 # unstable

        # System noise power (variance) parameters
        self.W = W # V[w_t] = V[x_1]
        self.V = V # V[v_t]

        # LQG parameters
        self.Q = Q
        self.R = R
        self.F = F

        # Pre-evaluate L(t) in the right order to avoid blowing the stack
        for t in range(self.T, 0, -1):
            self.L(t)

    def setRates(self, KC, KS):
        if (KC, KS) not in [(1, 1), (2, 1)]:
            raise ValueError("{{KC = {}; KS = {}}} unsupported".format(KC, KS))
        else:
            self.KC = KC
            self.KS = KS

    def setAnalog(self, SNR):
        self.SNR = SNR # 1 / V[n_t]
        self.analog = True

    def setDigital(self, n_codewords):
        self.n_codewords = n_codewords
        self.analog = False

    def setScheme(self, scheme):
        self.scheme = scheme
        if scheme == 'joint':
            assert self.analog
            if (self.KC, self.KS) == (1, 1):
                # Channel code signal-distortion ratio, 1 / V[neff_t]
                self.SDR0 = self.SNR # Optimum is achieved
            elif (self.KC, self.KS) == (2, 1):
                raise NotImplementedError("Spiral not implemented yet")

        elif scheme == 'separate':
            assert self.analog
            raise NotImplementedError("Tree codes not implemented yet")

        elif scheme == 'lloyd-max':
            assert not self.analog
            pass # Nothing to do

        else:
            raise ValueError("Unrecognized scheme: {}".format(scheme))


    def all(self):
        return {k: v for k, v in inspect.getmembers(self)
                if not k.startswith('__')}

    # Statically known parameters computed recursively using memoization

    @memoized
    def Pt(self, t, t_obs):
        if (t, t_obs) == (1, 0):
            return self.W
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
            return self.W
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
                np.polynomial.polynomial.Polynomial([
                    -self.Q * self.R,
                    -(self.Q + (self.alpha**2 - 1) * self.R),
                    1
                    ]).roots()))

    @memoized
    def Pt_inf(self):
        return next(filter(lambda S: S > 0,
                np.polynomial.polynomial.Polynomial([
                    -self.V * self.W,
                    -((self.alpha**2 - 1) * self.V + self.W),
                    1
                    ]).roots()))

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
