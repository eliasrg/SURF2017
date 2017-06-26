class TrivialEncoder:
    def encode(self, y):
        return (y,)

class TrivialDecoder:
    def decode(self, *code):
        assert(len(code) == 1)
        return code[0]


class KalmanFilter:
    def __init__(self, estimate, variance):
        self.estimate = estimate
        self.variance = variance

    def predict(self, alpha, W, u):
        x_est, P = self.estimate, self.variance
        self.estimate = alpha * x_est + u
        self.variance = alpha**2 * P + W

    def update(self, V, y):
        x_est, P = self.estimate, self.variance
        K = P / (P + V)
        self.estimate = x_est + K * (y - x_est)
        self.variance = K * V
