from .node import Node
from utilities import hamming_distance

import numpy as np
from queue import PriorityQueue
from numbers import Number

class StackDecoder:
    """Decodes a convolutional code transmitted over a BSC.
    Designed so that decoding incrementally by calling decode on a prefix
    of the received code sequence is efficient."""
    def __init__(self, code, p=None, SNR=None, bias_mode='R'):
        """If p is given, assumes BSC(p). If SNR is given, assumes AWGN(SNR)."""
        self.code = code
        if p is not None:
            self.p = p
        elif SNR is not None:
            self.SNR = SNR
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

        self.nodes = PriorityQueue()
        root = StackDecoder.Node(self.code)
        root.metric = 0
        self.nodes.put(root)

        # The first node in each layer
        self.first_nodes = [root]

    def extend(self, node, codeword):
        for child in node.extend():
            # Calculate the metric increment
            k = self.code.k
            n = self.code.n
            B = self.bias

            if hasattr(self, 'p'):
                # Binary codewords
                assert all(z in [0,1] for z in codeword.flatten())
                p = self.p
                d = hamming_distance(codeword, child.codeword)
                metric_increment = \
                        d * np.log2(p) + (n - d) * np.log2(1 - p) \
                        + (1 - B) * n
            else:
                # Real-valued codewords
                assert all(isinstance(z, float) for z in codeword.flatten())
                SNR = self.SNR

                # log2( w(zi|ci) / p(zi) )
                log_term = lambda z, c: \
                    1 - SNR/2 * (z - (-1)**c)**2 \
                    - np.log2(np.exp(-SNR/2 * (z - 1)**2)
                            + np.exp(-SNR/2 * (z + 1)**2))

                metric_increment = sum(log_term(z, c) - B
                        for z, c in zip(codeword, child.codeword))

            child.metric = node.metric + metric_increment

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

    def E0(self, rho):
        # Compared with (3b) of the tree code paper, 1 = log 2 and the summation
        # have been simplified away
        p = self.p
        return rho - (1 + rho) * np.log2(p**(1/(1+rho)) + (1 - p)**(1/(1+rho)))

    class Node(Node):
        """A node with a comparison operator for use in a min-priority queue."""
        def __lt__(self, other):
            return self.metric > other.metric
