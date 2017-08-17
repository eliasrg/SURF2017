from . import noisy_lloyd_max

class Encoder:
    def __init__(self, sim, tracker, convolutional_code):
        self.sim = sim
        self.noisy_lloyd_max_encoder = \
                noisy_lloyd_max.Encoder(sim, tracker, convolutional_code)

    def encode(self, *msg):
        binary_codeword = self.noisy_lloyd_max_encoder.encode(*msg)

        # Do trivial 2-PAM
        assert self.noisy_lloyd_max_encoder.convolutional_code.n \
                == self.sim.params.KC

        return [(-1)**bit for bit in binary_codeword]

Decoder = noisy_lloyd_max.Decoder
