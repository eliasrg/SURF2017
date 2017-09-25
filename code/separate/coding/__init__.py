# Copyright (c) 2017 Elias Riedel Gårding
# Licensed under the MIT License

from . import noisy_lloyd_max

class Encoder:
    def __init__(self, sim, tracker, convolutional_code, PAM=None):
        self.sim = sim
        self.noisy_lloyd_max_encoder = \
                noisy_lloyd_max.Encoder(sim, tracker, convolutional_code)
        if PAM is not None:
            self.PAM = PAM

    def encode(self, *msg):
        binary_codeword = self.noisy_lloyd_max_encoder.encode(*msg)

        if hasattr(self, 'PAM'):
            return self.PAM.modulate(binary_codeword)
        else:
            # Do trivial 2-PAM
            assert self.noisy_lloyd_max_encoder.convolutional_code.n \
                    == self.sim.params.KC

            return [(-1)**bit for bit in binary_codeword]

    def get_tracker(self):
        return self.noisy_lloyd_max_encoder.get_tracker()

    def get_bits_history(self):
        return self.noisy_lloyd_max_encoder.get_bits_history()

    def get_convolutional_code(self):
        return self.noisy_lloyd_max_encoder.get_convolutional_code()

Decoder = noisy_lloyd_max.Decoder
