# Copyright (c) 2017 Elias Riedel Gårding
# Licensed under the MIT License

from utilities import to_column_vector, int_to_bits, bits_to_int

import numpy as np


class Constellation:
    """A mapping m: ℤ2^n → ℝ^K."""
    @classmethod
    def uniform(cls, n):
        "Equidistant points (K = 1)."

        # {-(2^n - 1), -(2^n - 3), ..., 2^n - 3, 2^n - 1} (2^n integers)
        ints = 2 * np.arange(2**n) - (2**n - 1)

        return cls(n, 1, [np.array([x]) for x in ints])

    @classmethod
    def cartesian_product(cls, *constellations, repeat=None):
        # Cartesian power
        if repeat is not None:
            (constellation,) = constellations
            return cls.cartesian_product(*(repeat * [constellation]))

        if len(constellations) == 1:
            return constellations[0]
        else:
            last = constellations[-1]
            init = constellations[:-1]
            inner = cls.cartesian_product(*init)

            points = [np.concatenate((p, q))
                    for p in inner.points for q in last.points]

            return cls(inner.n + last.n, inner.K + last.K, points)

    def __init__(self, n, K, points):
        for p in points:
            if type(p) != np.ndarray:
                raise ValueError(
                        "Point not of type numpy.ndarray: {}" .format(p))

        self.n = n
        self.K = K
        self.points = points

    def modulate(self, bits):
        return list(self.points[bits_to_int(bits)])

    def demodulate(self, point):
        index = min(range(2**self.n),
                key=lambda i: norm_sq(point - self.points[i]))
        return int_to_bits(index, self.n)

    def metric_increment(self, SNR, bias, received, codeword):
        # Real-valued codewords
        assert all(isinstance(z, float) for z in received.flatten())

        return (1 - bias) * self.n \
                - SNR / (2 * np.log(2)) \
                    * norm_sq(received -
                            to_column_vector(self.modulate(codeword))) \
                - np.log2(sum(np.exp(-SNR/2 * norm_sq(
                                    received - to_column_vector(point)))
                              for point in self.points))

    def normalize(self, new_power=1):
        """Normalize so that the average power is 1."""
        power = np.mean([norm_sq(p) for p in self.points])
        factor = np.sqrt(new_power / power)
        new_points = [factor * p for p in self.points]

        return self.__class__(self.n, self.K, new_points)


def norm_sq(x):
    return np.sum(x**2)
