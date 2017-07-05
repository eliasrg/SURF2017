class Plant:
    """The system {x(t+1) = a x(t) + w(t) + u(t); y(t) = x(t) + v(t)}."""
    def __init__(self, alpha, draw_x1, draw_w, draw_v):
        self.alpha = alpha
        self.draw_x1 = draw_x1
        self.draw_w = draw_w
        self.draw_v = draw_v

        self.x = draw_x1()
        self.y = self.x + self.draw_v()

    def step(self, u):
        self.x = self.alpha * self.x + self.draw_w() + u
        self.y = self.x + self.draw_v()


class Channel:
    def __init__(self, draw_n):
        self.draw_n = draw_n
        self.total_power = 0
        self.uses = 0

    def transmit(self, msg):
        for a in msg:
            self.total_power += a**2
        self.uses += 1

        return (a + self.draw_n() for a in msg)

    def average_power(self):
        return self.total_power / self.uses


class LQGCost:
    """A cost function of the form
    J(T) = 1/T (F x(T+1)² + Σ{t = 1..t}(Q x(t)² + R u(t)²))."""
    def __init__(self, plant, Q, R, F):
        self.plant = plant
        self.Q = Q
        self.R = R
        self.F = F

        self.x_sum = 0.0
        self.u_sum = 0.0
        self.last_x = plant.x

    def step(self, u):
        """To be called immediately after plant.step()"""
        self.x_sum += self.Q * self.last_x**2
        self.u_sum += self.R * u**2
        self.last_x = self.plant.x

    def evaluate(self, t):
        """If called at time t (i.e. after t-1 steps), will return J(t-1).

    The parameter t must agree with the current simulation time."""
        if t <= 1:
            return float('nan')
        else:
            return (self.x_sum + self.u_sum + self.F * self.last_x**2) / (t - 1)
