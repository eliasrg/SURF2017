import sys, os
sys.path.insert(0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'code')))

import unittest
import numpy as np
from separate.coding.channel.convolutional \
        import ConvolutionalCode, ViterbiDecoder, StackDecoder

class TestConvolutionalEncoder(unittest.TestCase):
    """Test of convolutional encoder (from a homework problem)."""

    Gs = [
        np.array([[1, 1, 1]]).transpose(),
        np.array([[0, 0, 1]]).transpose(),
        np.array([[0, 1, 0]]).transpose(),
        np.array([[0, 1, 0]]).transpose(),
        np.array([[0, 0, 1]]).transpose(),
        np.array([[0, 1, 1]]).transpose()]
    code = ConvolutionalCode(3, 1, Gs)
    msg = [np.array([[x]]) for x in [1,1,1,0,1,0,0,1,0,0,0,0,0]]
    nominal_code = np.array([1,1,1, 1,1,0, 1,0,0, 0,0,1, 1,1,0, 0,0,1, 0,0,0,
                             1,1,0, 0,0,0, 0,0,1, 0,1,0, 0,0,1, 0,1,1])

    def test_encode(self):
        code = self.code.encode_sequence(self.msg)
        self.assertTrue((np.array(code).flatten() == self.nominal_code).all())

class TestViterbiDecoder(unittest.TestCase):
    """Test of the Viterbi algorithm (from a homework problem)."""

    Gs = [
        np.array([[1, 1]]).transpose(),
        np.array([[0, 1]]).transpose(),
        np.array([[1, 1]]).transpose()]
    code = ConvolutionalCode(2, 1, Gs)
    received_sequence = [np.array([x]).transpose() for x in
            [[1,0], [1,1], [1,0], [1,1], [1,0], [1,1], [1,0]]]
    decoder = ViterbiDecoder(code)
    nominal_input = [np.array([[x]]) for x in [0, 1, 1, 1, 0, 0, 0]]

    def test_decode(self):
        decoder = self.decoder
        decoder.decode(self.received_sequence)
        # In the homework problem, it was known that the last two bits were 0.
        decoded_inputs = [history for history in map(list, decoder.best_inputs)
                if history[-2:] == [np.array([[0]])] * 2]

        self.assertTrue(all(node.depth == len(self.nominal_input)
            for node in decoder.best_nodes))
        self.assertEqual(decoder.best_hamming_distance, 3)
        self.assertEqual(decoded_inputs, [self.nominal_input])

class TestStackDecoder(unittest.TestCase):
    """Test of the Stack algorithm (from Viterbi and Omura, Figure 6.1)."""

    Gs = [
        np.array([[1, 1]]).transpose(),
        np.array([[1, 0]]).transpose(),
        np.array([[1, 1]]).transpose()]
    code = ConvolutionalCode(2, 1, Gs)
    received_sequence = [np.array([x]).transpose() for x in
            [[0,1], [1,0], [0,1], [1,0], [1,1]]]
    decoder = StackDecoder(code, 0.03)
    nominal_input = [np.array([[x]]) for x in [1, 0, 1, 0, 0]]

    def test_decode(self):
        decoded_input = list(self.decoder.decode(self.received_sequence))
        self.assertEqual(decoded_input, self.nominal_input)

if __name__ == '__main__':
    unittest.main()
