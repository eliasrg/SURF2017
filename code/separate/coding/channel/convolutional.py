class ConvolutionalCode:
    def __init__(self, n, k, Gs):
        self.n = n # Output block length
        self.k = k # Input block length
        self.Gs = Gs # Generator matrices G_1, ..., G_T ∊ ℤ2^(n×k)
        assert(all(G.shape == (n,k) for G in Gs))

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
    data is a list of column vectors in ℤ2^k."""
        data = list(data)
        return [self.encode(data[:i+1]) for i in range(len(data))]
