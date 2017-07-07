class Observer:
    """Observes the plant and transmits the observation without preprocessing.
    Note: Currently violates the power constraint."""
    def __init__(self, sim):
        self.sim = sim

    def observe(self, t, y):
        return (y,)

class Controller:
    """Given a single observation of the plant's state, constructs the control
    signal to drive it to zero as quickly as possible without concern for the
    amplitude of the control signal."""
    def control(self, t, *msg):
        assert(len(msg) == 1)
        x_est = msg[0]
        return -self.sim.params.alpha * x_est
