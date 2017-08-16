from math import atan2, floor, tau
import numpy as np
from numpy import exp
from scipy.optimize import newton

class SpiralMap:
    # Use a complex representation internally for simplicity
    def __init__(self, omega, c):
        self.omega = omega
        self.c = c

    def encode_complex(self, s):
        return self.c * s * exp(1j *  self.omega * abs(s))

    def encode(self, s):
        z = self.encode_complex(s)
        return np.array([z.real, z.imag])

    def decode(self, received):
        """ML (min-distance) decoding."""
        c = self.c
        omega = self.omega
        b = complex(received[0], received[1])
        b_conj = b.conjugate()
        b_dist = lambda s: abs(self.encode_complex(s) - b)

        def s_estimates(sign):
            angle_guess = tau * floor(omega * abs(b) / (tau * c)) \
                    + atan2(sign * b.imag, sign * b.real) % tau
            # Just sprinkle a few guesses around
            s_guesses = [(angle_guess + n*tau/4) / (sign * omega)
                for n in range(-4, 5)]

            iσω = 1j * sign * omega
            def g(s):
                return (b_conj * exp(iσω * s) - c * s) * (1 + iσω * s)

            def Dg(s):
                return (iσω * b_conj * exp(iσω * s) - c) * (1 + iσω * s) \
                     + iσω * (b_conj * exp(iσω * s) - c * s)

            def f(s): return g(s).real
            def Df(s): return Dg(s).real

            # # f(s) is the scalar procuct (b - encode(s)) ∙ encode'(s)
            # # up to a factor of c. More straightforward definition:
            # def D(f): # Numerical differentiation
            #     h = 1e-12
            #     return lambda x: (f(x + h) - f(x)) / h
            # def f(s): return ( (b - self.encode_complex(s)).conjugate()
            #         * D(self.encode_complex)(s) ).real
            # Df = D(f)

            estimates = []
            for guess in s_guesses:
                try:
                    estimates.append(newton(f, guess, Df))
                except RuntimeError:
                    # Newton's method failed to converge
                    pass

            # Remove estimates with the wrong sign
            estimates = [s for s in estimates if s * sign >= 0]

            return estimates

        return min(s_estimates(1) + s_estimates(-1), key=b_dist)
