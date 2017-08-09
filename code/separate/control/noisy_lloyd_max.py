class Observer:
    """Calculates an estimate of the plant state as if the channel
    were noiseless and transmits it to the controller."""
    def __init__(self, sim):
        self.sim = sim
        self.x_u_ideal = DeterministicPlant(sim.alpha)
        self.x_u_actual = DeterministicPlant(sim.alpha)

    def observe(self, t, y):
        # Update the controlled part of the actual plant
        u_actual_previous = self.sim.globals.u[t-1] if t > 1 else 0
        self.x_u_actual.step(u_actual_previous)

        # Calculate the ideal (noiseless) estimate
        x_est = y # TODO take observation noise into account
        x_est_uncontrolled = x_est - self.x_u_actual.value
        x_est_ideal = x_est_uncontrolled + self.x_u_ideal.value

        # Update the controlled part of the ideal plant
        u_ideal = -sim.params.L(t) * x_est_ideal
        self.x_u_ideal.step(u_ideal)

        # Pass to the source and channel encoders
        return (x_est_ideal,)


class Controller:
    def __init__(self, sim):
        self.sim = sim
        self.u_history = []

    def control(self, t, *msg):
        sim = self.sim

        # Decode the state estimate
        assert len(msg) == 2 # One real number, one ??? to indicate mistake
        x_est, mistake = msg

        # Generate the control signal
        u = -sim.params.L(t) * x_est

        # Correct for mistake
        if mistake:
            pass # TODO

        return u


class DeterministicPlant:
    """The deterministic plant x(0) = 0; x(t+1) = α x(t) + u(t)."""
    def __init__(self, alpha):
        self.value = 0
        self.alpha = alpha

    def step(self, u):
        self.value = self.alpha * self.value + u
