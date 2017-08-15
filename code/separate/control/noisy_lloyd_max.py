class Observer:
    """Calculates an estimate of the plant state as if the channel
    were noiseless and transmits it to the controller."""
    def __init__(self, sim):
        self.sim = sim
        self.x_u_ideal = DeterministicPlant(sim.params.alpha)
        self.x_u_actual = DeterministicPlant(sim.params.alpha)

    def observe(self, t, y):
        # Update the controlled part of the actual plant
        u_actual_previous = self.sim.globals.u[t-1] if t > 1 else 0
        self.x_u_actual.step(u_actual_previous)

        # Calculate the ideal (noiseless) estimate
        x_est = y # TODO take observation noise into account
        x_est_uncontrolled = x_est - self.x_u_actual.value
        x_est_ideal = x_est_uncontrolled + self.x_u_ideal.value

        # Quantize it TODO restructure so quantization only happens once?
        tracker = self.sim.encoder.source_encoder.tracker
        x_est_ideal_quantized = \
                tracker.lm_decoder.decode(tracker.lm_encoder.encode(x_est))

        # Update the controlled part of the ideal plant
        u_ideal = -self.sim.params.L(t) * x_est_ideal_quantized
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
        # One real number and None or a list to indicate mistake
        assert len(msg) == 2
        x_est, revised_x_est_history = msg

        # Generate the control signal
        u = -sim.params.L(t) * x_est
        self.u_history.append(u)

        # Correct for mistake
        if revised_x_est_history is not None:
            x_correction = DeterministicPlant(sim.params.alpha)
            n_revised_steps = len(revised_x_est_history)

            print("Mistake: n_revised_steps = {}\nrevised_x_est_history = {}"
                    .format(n_revised_steps, revised_x_est_history))

            # First time where there is a difference between estimates
            t1 = t - n_revised_steps

            # Calculate the deviation -x_correction due to the mistake
            for tau in range(t1, t):
                u_actual = self.u_history[tau - 1] # Time starts at 1
                x_corrected = revised_x_est_history[tau - t1]
                u_corrected = -sim.params.L(tau) * x_corrected
                x_correction.step(u_corrected - u_actual)

                # Rewrite the output history as if no mistake had occurred
                self.u_history[tau - 1] = u_corrected

            u += sim.params.alpha * x_correction.value
            # Do not record this correction in self.u_history[t].
            # We have already recorded that the ideal control signal was sent.

        return u


class DeterministicPlant:
    """The deterministic plant x(0) = 0; x(t+1) = α x(t) + u(t)."""
    def __init__(self, alpha):
        self.value = 0
        self.alpha = alpha

    def step(self, u):
        self.value = self.alpha * self.value + u
