from math import sqrt

class TrivialEncoder:
    def encode(self, sim, t, y):
        return (y,)

class TrivialDecoder:
    def decode(self, sim, t, *code):
        assert(len(code) == 1)
        x_est = code[0]
        return -sim.params.alpha * x_est


class Encoder:
    def __init__(self):
        pass

    def encode(self, sim, t, y):
        # Kalman filter prediction of x(t)
        self.x_est = (0 if t == 1
                  else sim.params.alpha * self.x_est + sim.globals.u[t-1])

        # Update Kalman filter
        self.x_est += sim.params.Kt(t) * (y - self.x_est) # (15)

        # Generate the error signal (8a)
        s = self.x_est - sim.globals.x_est_r[t, t-1]

        # Normalize it (9)
        s_norm = s / sqrt(sim.params.Pr(t, t-1) - sim.params.Pt(t, t))

        # Send with trivial coding
        return (s_norm,)


class Decoder:
    def __init__(self, sim):
        # Initialize MMSE estimate
        self.x_est = 0
        sim.globals.x_est_r[1, 0] = self.x_est

    def decode(self, sim, t, *code):
        # Trivial decoding of s_norm_est
        assert(len(code) == 1)
        s_norm_est = code[0]

        # Unnormalize (10a)
        s_est = sqrt(sim.params.Pr(t, t-1) - sim.params.Pt(t, t)) * s_norm_est

        # Construct the estimate of x(t) (11)
        self.x_est += sim.params.SDR0 / (1 + sim.params.SDR0) * s_est
        sim.globals.x_est_r[t, t] = self.x_est

        # Generate the control signal (below (12))
        u = -sim.params.L(t) * self.x_est

        # Generate the prediction for x(t+1) (above (13a))
        self.x_est = sim.params.alpha * self.x_est + u
        sim.globals.x_est_r[t+1, t] = self.x_est

        return u
