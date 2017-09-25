# Copyright (c) 2017 Elias Riedel Gårding
# Licensed under the MIT License

from utilities import to_column_vector

import numpy as np


class Constellation:
    """A mapping m: ℤ2^n → ℝ^K."""
    def __init__(self, n, K, points):
        self.n = n
        self.K = K
        self.points = points

    def modulate(self, bits):
        return self.points[bits_to_int(bits)]

    def demodulate(self, point):
        index = min(range(2**self.n),
                key=lambda i: norm_sq(point - self.points[i]))
        return int_to_bits(index, self.n)

    def metric_increment(self, SNR, bias, received, codeword):
        # Real-valued codewords
        assert all(isinstance(z, float) for z in received.flatten())

        return (1 - bias) * self.n \
                - SNR / (2 * np.log(2)) \
                    * norm_sq(received - self.modulate(codeword)) \
                - np.log2(sum(np.exp(-SNR/2 * norm_sq(received - point))
                              for point in self.points))


def int_to_bits(i, n):
    bits = []
    while i != 0:
        bits.insert(0, i & 1)
        i >>= 1
    return to_column_vector([0] * (n - len(bits)) + bits)


def bits_to_int(bits):
    i = 0
    for bit in bits.flatten():
        i = i << 1 | bit
    return i

def norm_sq(x):
    return np.sum(x**2)
