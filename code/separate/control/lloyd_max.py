class Observer:
    """The observer/transmitter described in the Lloyd-Max paper."""
    def __init__(self, sim):
        self.sim = sim

    def observe(self, t, y):
        # Pass to the (source) encoder
        return (y,)


class Controller:
    """The controller/receiver described in the Lloyd-Max paper."""
    def __init__(self, sim):
        self.sim = sim

    def control(self, t, *msg):
        sim = self.sim

        # Decode the state estimate
        assert len(msg) == 1 # One real number
        x_est = msg[0]

        # Generate the control signal
        u = -sim.params.L(t) * x_est

        return u
