from utilities import full_rank

import numpy as np
import scipy.stats as stats

class ConvolutionalCode:
    def __init__(self, n, k, Gs):
        self.n = n # Output block length
        self.k = k # Input block length
        self.Gs = Gs # Generator matrices G_1, ..., G_T ∊ ℤ2^(n×k)
        assert all(G.shape == (n,k) for G in Gs)

    def encode(self, data):
        """Performs one encoding step.
    data is a sequence of column vectors in ℤ2^k, in chronological order."""
        return self.encode_reversed(reversed(list(data)))

    def encode_reversed(self, reversed_data):
        """Performs one encoding step.
    data is a sequence of column vectors in ℤ2^k,
    in reverse chronological order."""
        return sum(G @ b for G, b in zip(self.Gs, reversed_data)) % 2

    def encode_sequence(self, data):
        """Encodes an entire sequence.
    data is a sequence of column vectors in ℤ2^k."""
        data = list(data)
        return [self.encode(data[:i+1]) for i in range(len(data))]

    def rate(self):
        return self.k / self.n

    def constraint_length(self):
        """Returns the constraint length (maximum delay) of the code."""
        # Assumes that the Gs[-1] is nonzero
        return len(self.Gs) - 1

    def generator_matrix(self, n_blocks):
        """The ((n n_blocks) × (k n_blocks)) generator matrix for the code."""
        # Taken from A. Khina's code, modified to support n_blocks < len(Gs)
        k, n = self.k, self.n
        G = np.zeros([n * n_blocks, k * n_blocks])
        for i, g in enumerate(self.Gs[:n_blocks]):
            G[i*n:, :(n_blocks - i) * k] += np.kron(np.eye(n_blocks - i), g)

        return G.astype(int)

    @staticmethod
    def random_code(n, k, constraint_length):
        """Create a random code drawn from the LTI ensemble."""
        random_matrix = lambda: stats.bernoulli.rvs(0.5, size=[n, k])

        Gs = [random_matrix() for _ in range(constraint_length + 1)]

        # Force G0 be a matrix of full rank
        while not full_rank(Gs[0]):
            Gs[0] = random_matrix()

        return ConvolutionalCode(n, k, Gs)
