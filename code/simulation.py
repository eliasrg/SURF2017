from utilities import memoized

def simulate(plant, channel, encoder, decoder, LQG, T):
    # TODO Initialize DP evaluator and pass it to encoder and decoder
    t = 1
    yield t
    while t < T:
        x_est = decoder.decode(t,
                *(channel.transmit(p) for p in encoder.encode(t, plant.y)))
        u = -plant.alpha * x_est
        plant.step(u)
        LQG.step(u)
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
