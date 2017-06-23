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
