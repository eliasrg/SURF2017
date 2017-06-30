import numpy as np
from math import sqrt
from typing import NewType, Callable

Sampler = NewType('Sampler', Callable[[], float])

def onepoint(x) -> Sampler:
    return lambda: x

zero = onepoint(0)

def gaussian(var: float) -> Sampler:
    stddev = sqrt(var)
    return lambda: np.random.normal(0, stddev)
