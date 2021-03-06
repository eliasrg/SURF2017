# Copyright (c) 2017 Elias Riedel Gårding
# Licensed under the MIT License

from . import source
from .convolutional import stack
from utilities import to_column_vector, int_to_bits, bits_to_int

import numpy as np
import warnings

class Encoder:
    def __init__(self, sim, tracker, convolutional_code):
        self.sim = sim
        self.source_encoder = source.Encoder(sim, tracker)
        self.convolutional_code = convolutional_code
        self.bits_history = []

    def encode(self, *msg):
        # Quantize
        i = self.source_encoder.encode(*msg)[0]

        # Convert quantization index to bits
        # TODO Worth doing in multiple ways in the digital case?
        bits = int_to_bits(i, self.convolutional_code.k)
        self.bits_history.append(bits)

        # Encode bits with tree code
        codeword = self.convolutional_code.encode(self.bits_history)

        # Send to channel
        return codeword.flatten()

    def get_tracker(self):
        return self.source_encoder.tracker

    def get_bits_history(self):
        return self.bits_history

    def get_convolutional_code(self):
        return self.convolutional_code

class Decoder:
    """Combined source and channel decoder."""
    def __init__(self, sim, tracker, convolutional_code):
        self.sim = sim
        self.source_decoder = source.Decoder(sim, tracker)
        self.source_decoder_history = [self.source_decoder]
        self.convolutional_code = convolutional_code
        self.received_codeword_history = []

        if self.sim.params.analog:
            # AWGN(SNR)
            self.stack_decoder = stack.StackDecoder(
                    self.convolutional_code, SNR=self.sim.params.SNR,
                    PAM=getattr(self.sim.params, 'PAM', None))
        else:
            # BSC(p)
            self.stack_decoder = stack.StackDecoder(
                    self.convolutional_code, p=self.sim.params.p)

        self.error_exponent_sanity_check()

    def decode(self, *msg):
        """The first return value is the decoded plant state estimate.
        The second return value is None if no previous decoding mistake was
        detected and a list of the revised estimates of the state at previous
        times (not including the current time, as this is the last return
        value)."""
        received_codeword = to_column_vector(msg)
        self.received_codeword_history.append(received_codeword)

        # Decode using stack decoder
        bits = self.stack_decoder.decode_block(self.received_codeword_history)

        # Check for mistake
        revised_x_est_history = None

        # At least one branch has been made
        assert len(self.stack_decoder.first_nodes) >= 2

        old_node, new_node = self.stack_decoder.first_nodes[-2:]
        if new_node.parent is not old_node: # Mistake detected
            # Recover corrected quantizer index history
            ancestor = new_node.first_common_ancestor(old_node)
            revised_i_history = [bits_to_int(bits)
                    for bits in new_node.parent.input_history(stop_at=ancestor)]

            # Delete old source decoder history
            n_revised_steps = len(revised_i_history)
            assert n_revised_steps > 0
            del self.source_decoder_history[len(self.source_decoder_history)
                    - (n_revised_steps-1):]
            self.source_decoder = self.source_decoder_history[-1]
            del self.source_decoder_history[-1]

            # Update source decoder history and recover revised x estimates
            revised_x_est_history = [
                    self.source_decode(i) for i in revised_i_history]

        # Convert to quantization index
        i = bits_to_int(bits)

        # Recover quantized estimate of x
        x_est = self.source_decode(i)

        return x_est, revised_x_est_history

    def source_decode(self, i):
        self.source_decoder_history.append(self.source_decoder.clone())
        return self.source_decoder.decode(i)[0]

    def error_exponent_sanity_check(self):
        EJ = self.stack_decoder.EJ()
        alpha = self.sim.params.alpha
        if EJ < 2 * np.log2(alpha):
            warnings.warn((
                "EJ = {:.4f} is less than 2 log2 α = {:.4f}. "
                + "Expect control to fail."
                ).format(EJ, 2 * np.log2(alpha)), RuntimeWarning)
