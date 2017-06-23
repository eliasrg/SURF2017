import numpy as np
from typing import NewType, Callable

Sampler = NewType('Sampler', Callable[[], float])

def onepoint(x) -> Sampler:
    return lambda: x

zero = onepoint(0)

def gaussian(var: float) -> Sampler:
    return lambda: np.random.normal(0, var)
