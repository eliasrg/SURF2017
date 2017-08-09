from . import source
from .convolutional import stack
from utilities import to_column_vector

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
        bits = int_to_bits(i)
        self.bits_history.append(bits)

        # Encode bits with tree code
        codeword = self.convolutional_code.encode(self.bits_history)

        # Send to channel
        return bits

class Decoder:
    """Combined source and channel encoder."""
    def __init__(self, sim, tracker, convolutional_code):
        self.sim = sim
        self.source_decoder = source.Decoder(sim, tracker)
        self.source_decoder_history = [self.source_decoder]
        self.convolutional_code = convolutional_code
        self.received_codeword_history = []
        self.stack_decoder = stack.StackDecoder(
                self.convolutional_code, self.sim.channel.p) # Assume BSC

    def decode(self, *msg):
        received_codeword = to_column_vector(msg)
        self.received_codeword_history.append(received_codeword)

        # Decode using stack decoder
        bits = self.stack_decoder.decode_block(self.received_codeword_history)

        # Convert to quantization index
        i = bits_to_int(bits)

        # Recover quantized estimate of x
        x_est = self.source_decoder.decode(i)[0]
        self.source_decoder_history.append(self.source_decoder.clone())

        # TODO Check for mistake
        mistake = None

        return x_est, mistake


def int_to_bits(i):
    assert i in [0,1] # Assume k = 1 in line with 2-PAM
    return to_column_vector([i])


def bits_to_int(bits):
    assert len(bits) == 1 # Assume k = 1 in line with 2-PAM
    i = bits.flatten()[0]
    assert i in [0,1]
    return i
