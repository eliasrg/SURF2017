from distributions import onepoint, zero, gaussian
from system import Plant, Channel, LQGCost
from coding import TrivialEncoder, TrivialDecoder
from utilities import memoized

class Simulation:
    def __init__(self, params):
        self.params = params

        self.plant = Plant(params.alpha, gaussian(params.P0),
                gaussian(params.W), gaussian(params.V))
        self.channel = Channel(gaussian(1 / params.SNR))
        self.encoder = TrivialEncoder()
        self.decoder = TrivialDecoder()

        self.LQG = LQGCost(self.plant, params.Q, params.R, params.F)

    def simulate(self, T):
        # TODO Initialize DP evaluator and pass it to encoder and decoder
        t = 1
        yield t
        while t < T:
            x_est = self.decoder.decode(t,
                    *(self.channel.transmit(p)
                        for p in self.encoder.encode(t, self.plant.y)))
            u = -self.plant.alpha * x_est
            self.plant.step(u)
            self.LQG.step(u)
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

    @memoized
    def Pt(t, t_obs):
        if t == 1:
            return self.P0
        elif t_obs == t:
            return self.Kt(t) * self.V
        elif t_obs == t-1:
            return self.alpha**2 * self.Pt(t-1, t-1) + self.W

    @memoized
    def Kt(t):
        P = self.Pt(t, t-1)
        return P / (P + V)

    @memoized
    def Pr(t, t_obs):
        if False: # TODO Which is the right base case?
            pass
        elif t_obs == t:
            return (self.Pr(t, t-1) + self.SDR0 * self.Pt(t, t))
        elif t_obs == t-1:
            return self.alpha**2 * self.Pr(t-1, t-1) + W
