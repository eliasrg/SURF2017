# Copyright (c) 2017 Elias Riedel Gårding
# Licensed under the MIT License

from .node import Node
from utilities import hamming_distance, memoized
from distributions import gaussian

import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
from scipy.optimize import minimize
from queue import PriorityQueue
from numbers import Number
import warnings

class StackDecoder:
    """Decodes a convolutional code transmitted over a BSC.
    Designed so that decoding incrementally by calling decode on a prefix
    of the received code sequence is efficient."""
    def __init__(self, code, p=None, SNR=None, bias_mode='R'):
        """If p is given, assumes BSC(p). If SNR is given, assumes AWGN(SNR)."""
        self.code = code
        if p is not None:
            self.compute_metric_increment = BSC_metric_increment(code.n, p)
        elif SNR is not None:
            self.SNR = SNR
            self.compute_metric_increment = AWGN_2PAM_metric_increment(SNR)
        else:
            raise ValueError("p or SNR must be given")

        if bias_mode == 'R':
            self.bias = self.code.rate()
        elif bias_mode == 'E0':
            self.bias = self.E0(1)
        elif isinstance(bias_mode, Number):
            self.bias = bias_mode
        else:
            raise ValueError(
                    "{} is not 'R', 'E0' or a number".format(bias_mode))

        self.bias_sanity_check()

        self.nodes = PriorityQueue()
        root = StackDecoder.Node(self.code)
        root.metric = 0
        self.nodes.put(root)

        # The first node in each layer
        self.first_nodes = [root]

    def extend(self, node, received):
        for child in node.extend():
            child.metric = node.metric + self.compute_metric_increment(
                    self.bias, received, child.codeword)

            self.nodes.put(child)

    def decode_node(self, received_sequence):
        """Returns the node corresponding to the decoded path."""
        # Run until we reach the first full-length path
        while True:
            node = self.nodes.get()

            depth = len(self.first_nodes) - 1 # Max depth among explored nodes
            if node.depth == depth + 1:
                self.first_nodes.append(node)

            if node.depth == len(received_sequence):
                # Add it back to the queue so it can be extended in the future
                self.nodes.put(node)

                return node

            self.extend(node, received_sequence[node.depth])

    def decode(self, received_sequence):
        """Returns the decoded bit sequence."""
        return self.decode_node(received_sequence).input_history()

    def decode_block(self, received_sequence):
        """Returns the last block of the decoded bit sequence."""
        return self.decode_node(received_sequence).input_block

    def E0(self, rho, simple_bound=False):
        # Compared with (3b) of the tree code paper, 1 = log 2 and the summation
        # have been simplified away
        if hasattr(self, 'p') or simple_bound:
            if hasattr(self, 'p'):
                p = self.p
            else:
                # Lower bound on E0 (slicing, i.e. convert AWGN → BSC at a loss)
                # Bit flip (sign crossover) if noise is larger than 1
                p = gaussian(1 / self.SNR).sf(1)
            return rho - (1 + rho) * np.log2(
                    p**(1/(1+rho)) + (1 - p)**(1/(1+rho)))
        else:
            # From an example in the tree code paper
            w = gaussian(1 / self.SNR).pdf
            return 1 + rho - np.log2(quad(lambda z:
                    ( w(z - 1)**(1/(1+rho)) + w(z + 1)**(1/(1+rho)) )**(1+rho),
                    -np.inf, np.inf)[0])

    @memoized
    def EJ(self):
        # Function to maximize
        f = lambda rho: rho / (1 + rho) * (
                self.E0(rho) + self.bias - (1 + rho) * self.code.rate())
        return -minimize(lambda rho: -f(rho), 0.5, bounds=[[0,1]]).fun[0]

    def bias_sanity_check(self):
        E0 = self.E0(1)
        if self.bias > E0:
            warnings.warn((
                "Bias {:.4f} is larger than E0 = {:.4f}. "
                + "Expect high decoding time complexity."
                ).format(self.bias, E0), RuntimeWarning)

    class Node(Node):
        """A node with a comparison operator for use in a min-priority queue."""
        def __lt__(self, other):
            return self.metric > other.metric


def BSC_metric_increment(n, p):
    def metric_increment(bias, received, codeword):
        # Binary codewords
        assert all(z in [0,1] for z in received.flatten())

        d = hamming_distance(received, codeword)
        return d * np.log2(p) + (n - d) * np.log2(1 - p) \
               + (1 - bias) * n

    return metric_increment

def AWGN_2PAM_metric_increment(SNR):
    def metric_increment(bias, received, codeword):
        # Real-valued codewords
        assert all(isinstance(z, float) for z in received.flatten())

        # log2( w(zi|ci) / p(zi) )
        log_term = lambda z, c: \
            1 - SNR / (2 * np.log(2)) * (z - (-1)**c)**2 \
            - np.log2(np.exp(-SNR/2 * (z - 1)**2)
                    + np.exp(-SNR/2 * (z + 1)**2))

        return sum(log_term(z, c) - bias
                for z, c in zip(received, codeword))

    return metric_increment
