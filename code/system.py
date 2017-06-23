"""The system {x(t+1) = a x(t) + w(t) + u(t); y(t) = x(t) + v(t)}."""
class Plant:
    def __init__(self, a, draw_x0, draw_w, draw_v):
        self.a = a
        self.draw_x0 = draw_x0
        self.draw_w = draw_w
        self.draw_v = draw_v

        self.x = draw_x0()
        self.y = self.x + self.draw_v()

    def step(self, u):
        self.x = self.a * self.x + self.draw_w() + u
        self.y = self.x + self.draw_v()


class Channel:
    def __init__(self, draw_n):
        self.draw_n = draw_n

    def transmit(self, a):
        return a + self.draw_n()


"""A cost function of the form
    J(T) = 1/T (F x(T+1)² + Σ{t = 1..t}(Q x(t)² + R u(t)²))."""
class LQGCost:
    def __init__(self, plant, Q, R, F):
        self.plant = plant
        self.Q = Q
        self.R = R
        self.F = F

        self.x_sum = 0.0
        self.u_sum = 0.0
        self.last_x = plant.x

    """To be called immediately after plant.step()"""
    def step(self, u):
        self.x_sum += self.Q * self.last_x * self.last_x
        self.u_sum += self.R * u * u
        self.last_x = self.plant.x

    def evaluate(self):
        return self.x_sum + self.u_sum + self.F * self.last_x
