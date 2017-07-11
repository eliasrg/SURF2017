class Observer:
    """The observer/transmitter described in the Lloyd-Max paper."""
    def __init__(self, sim):
        self.sim = sim

    def observe(self, t, y):
        pass # TODO


class Controller:
    """The controller/receiver described in the Lloyd-Max paper."""
    def __init__(self, sim):
        self.sim = sim

    def control(self, t, *msg):
        sim = self.sim

        pass # TODO
