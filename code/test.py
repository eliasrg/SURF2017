import numpy as np
import itertools as it

from misc import onepoint, zero, gaussian
from system import Plant, Channel
from simulation import simulate
from coding import TrivialEncoder, TrivialDecoder

plant = Plant(2, gaussian(10000), gaussian(1), gaussian(1))
# plant = Plant(2, onepoint(10000), zero, zero)
channel = Channel(zero)
encoder = TrivialEncoder()
decoder = TrivialDecoder()

for _ in it.islice(simulate(plant, channel, encoder, decoder), 40):
    print(plant.y)
