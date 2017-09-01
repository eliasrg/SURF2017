# Copyright (c) 2017 Elias Riedel Gårding
# Licensed under the MIT License

import sys, os
sys.path.insert(0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'code')))

import unittest
import numpy as np
import scipy.stats as stats
from separate.coding.convolutional \
        import ConvolutionalCode, NaiveMLDecoder, StackDecoder
from utilities import blockify, to_column_vector
from system import BinarySymmetricChannel

import reference.tree_codes as anatoly

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

class TestNaiveMLDecoder(unittest.TestCase):
    """Test of ML decoding (from a homework problem on
    the Viterbi algorithm)."""

    Gs = [
        np.array([[1, 1]]).transpose(),
        np.array([[0, 1]]).transpose(),
        np.array([[1, 1]]).transpose()]
    code = ConvolutionalCode(2, 1, Gs)
    received_sequence = [np.array([x]).transpose() for x in
            [[1,0], [1,1], [1,0], [1,1], [1,0], [1,1], [1,0]]]
    decoder = NaiveMLDecoder(code)
    nominal_input = [np.array([[x]]) for x in [0, 1, 1, 1, 0, 0, 0]]

    def test_decode(self):
        decoder = self.decoder
        decoder.decode(self.received_sequence, save=True)
        # In the homework problem, it was known that the last two bits were 0.
        decoded_inputs = [history for history in map(list, decoder.best_inputs)
                if history[-2:] == [np.array([[0]])] * 2]

        self.assertTrue(all(node.depth == len(self.nominal_input)
            for node in decoder.best_nodes))
        self.assertEqual(decoder.best_hamming_distance, 3)
        self.assertEqual(decoded_inputs, [self.nominal_input])

    @staticmethod
    def decode_long_code(n_blocks=10, noise=False):
        n = 3
        k = 2
        if noise: p = 0.03
        code = ConvolutionalCode.random_code(n, k, n_blocks - 1)

        input_sequence = stats.bernoulli.rvs(0.5, size = k * n_blocks)
        code_sequence = code.encode_sequence(blockify(input_sequence, k))

        channel = BinarySymmetricChannel(p if noise else 0)
        received_sequence = [list(channel.transmit(c)) for c in code_sequence]

        print("Encoded: {}".format(np.array(code_sequence).flatten()))
        if noise:
            print("Receive: {}".format(np.array(received_sequence).flatten()))
            print("Noise:   {}".format(np.array(received_sequence).flatten()
                ^ np.array(code_sequence).flatten()))
        print("Input:   {}".format(input_sequence))

        decoded = np.array(list(
            NaiveMLDecoder(code).decode(received_sequence))).flatten()

        print("Decoded: {}".format(decoded))
        print("Success!" if (decoded == input_sequence).all() else "Failure!")

class TestStackDecoder(unittest.TestCase):
    """Test of the Stack algorithm (from Viterbi and Omura, Figure 6.1)."""

    Gs = [
        np.array([[1, 1]]).transpose(),
        np.array([[1, 0]]).transpose(),
        np.array([[1, 1]]).transpose()]
    code = ConvolutionalCode(2, 1, Gs)
    received_sequence = [np.array([x]).transpose() for x in
            [[0,1], [1,0], [0,1], [1,0], [1,1]]]
    nominal_input = [np.array([[x]]) for x in [1, 0, 1, 0, 0]]

    def __init__(self, *args):
        super().__init__(*args)
        self.decoder = StackDecoder(self.code, 0.03)

    def test_decode(self):
        decoded_input = list(self.decoder.decode(self.received_sequence))
        self.assertEqual(decoded_input, self.nominal_input)

class TestStackRandomCode(unittest.TestCase):
    N_DEFAULT = 3
    K_DEFAULT = 2

    def test_decode(self, n_blocks=1000, n=N_DEFAULT, k=K_DEFAULT,
            noise=False, incremental=False):
        p = 0.03 if noise else 1e-10
        bias_mode = 'E0'
        code = ConvolutionalCode.random_code(n, k, n_blocks - 1)

        input_sequence = stats.bernoulli.rvs(0.5, size = k * n_blocks)
        code_sequence = code.encode_sequence(blockify(input_sequence, k))

        channel = BinarySymmetricChannel(p if noise else 0)
        received_sequence = [np.array(list(channel.transmit(c)))
                for c in code_sequence]

        print("Encoded: {}".format(np.array(code_sequence).flatten()))
        if noise:
            print("Receive: {}".format(np.array(received_sequence).flatten()))
            print("Noise:   {}".format(np.array(received_sequence).flatten()
                ^ np.array(code_sequence).flatten()))
        print("Input:   {}".format(input_sequence))

        decoder = StackDecoder(code, p, bias_mode=bias_mode)
        if incremental:
            decoded = np.array([
                decoder.decode_block(received_sequence[:i+1])
                for i in range(len(received_sequence))]).flatten()
        else:
            decoded = np.array(list(
                decoder.decode(received_sequence))).flatten()

        print("Decoded: {}".format(decoded))
        print("Success!" if (decoded == input_sequence).all() else "Failure!")

        # For interactive debugging
        self.code = code
        self.input_sequence = input_sequence
        self.code_sequence = code_sequence
        self.received_sequence = received_sequence
        self.decoder = decoder
        self.decoded = decoded

class CompareStackDecoders(unittest.TestCase):
    N_DEFAULT = 3
    K_DEFAULT = 2

    def test_random_code(self, n=N_DEFAULT, k=K_DEFAULT, noise=False):
        n_blocks = 17
        p = 0.03 if noise else 1e-10
        bias_mode = 'E0'
        code = ConvolutionalCode.random_code(n, k, n_blocks - 1)

        input_sequence = stats.bernoulli.rvs(0.5, size = k * n_blocks)
        code_sequence = code.encode_sequence(blockify(input_sequence, k))

        channel = BinarySymmetricChannel(p if noise else 0)
        received_sequence = [np.array(list(channel.transmit(c)))
                for c in code_sequence]

        print("Encoded: {}".format(np.array(code_sequence).flatten()))
        if noise:
            print("Receive: {}".format(np.array(received_sequence).flatten()))
            print("Noise:   {}".format(np.array(received_sequence).flatten()
                ^ np.array(code_sequence).flatten()))
        print("Input:   {}".format(input_sequence))

        node = StackDecoder(code, p, bias_mode=bias_mode) \
            .decode_node(received_sequence)
        metric_own = node.metric
        decoded_own = np.array(list(node.input_history())).flatten()
        print("Decoded: {} ({})".format(decoded_own,
            "success" if (decoded_own == input_sequence).all() else "failure"))

        metric_anatoly, decoded_anatoly, _ = anatoly.stack_dec(
                np.matrix(to_column_vector(received_sequence)),
                np.matrix(code.generator_matrix(n_blocks)),
                k, n, n_blocks, p, bias=bias_mode)

        anatoly_success = \
                (np.array(decoded_anatoly).flatten() == input_sequence).all()
        print("Anatoly: {} ({})".format(np.array(decoded_anatoly).flatten(),
            "success" if anatoly_success else "failure"))

        if not anatoly_success:
            print("Diff:    {}".format(
                np.array(decoded_anatoly).flatten() ^ input_sequence))

        print("Elias metric:   {}".format(metric_own))
        print("Anatoly metric: {}".format(metric_anatoly))

        self.assertTrue(
                (decoded_own == np.array(decoded_anatoly).flatten()).all())


    @staticmethod
    def find_failure(n=N_DEFAULT, k=K_DEFAULT):
        while True:
            try:
                CompareStackDecoders().test_random_code(n, k)
                print()
            except AssertionError:
                break


if __name__ == '__main__':
    unittest.main()
