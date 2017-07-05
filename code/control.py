from math import sqrt

class TrivialObserver:
    """Observes the plant and transmits the observation without preprocessing.
    Note: Currently violates the power constraint."""
    def __init__(self, sim):
        self.sim = sim

    def observe(self, t, y):
        return (y,)

class TrivialController:
    """Given a single observation of the plant's state, constructs the control
    signal to drive it to zero as quickly as possible without concern for the
    amplitude of the control signal."""
    def control(self, t, *msg):
        assert(len(msg) == 1)
        x_est = msg[0]
        return -self.sim.params.alpha * x_est


class Observer:
    """The observer/transmitter described in the paper."""
    def __init__(self, sim):
        self.sim = sim

    def observe(self, t, y):
        sim = self.sim

        # Kalman filter prediction of x(t)
        self.x_est = (0 if t == 1
                  else sim.params.alpha * self.x_est + sim.globals.u[t-1])

        # Update Kalman filter
        self.x_est += sim.params.Kt(t) * (y - self.x_est) # (15)

        # Generate the error signal (8a)
        s = self.x_est - sim.globals.x_est_r[t, t-1]

        # Normalize it (9)
        s_norm = s / sqrt(sim.params.Pr(t, t-1) - sim.params.Pt(t, t))

        # Send to the encoder
        return (s_norm,)


class Controller:
    """The controller/receiver described in the paper."""
    def __init__(self, sim):
        self.sim = sim

        # Initialize MMSE estimate
        self.x_est = 0
        sim.globals.x_est_r[1, 0] = self.x_est

    def control(self, t, *msg):
        sim = self.sim

        # Receive s_norm_est from decoder
        assert(len(msg) == 1)
        s_norm_est = msg[0]

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
