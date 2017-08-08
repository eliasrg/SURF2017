from .node import Node
from utilities import hamming_distance

import numpy as np
from queue import PriorityQueue
from numbers import Number

class StackDecoder:
    """Decodes a convolutional code transmitted over a BSC.
    Designed so that decoding incrementally by calling decode on a prefix
    of the received code sequence is efficient."""
    def __init__(self, code, p, bias_mode='R'):
        """Assumes binary symmetric channel with error probability p."""
        self.code = code
        self.p = p

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

    def extend(self, node, codeword):
        for child in node.extend():
            # Calculate the metric increment
            p = self.p
            k = self.code.k
            n = self.code.n
            B = self.bias
            d = hamming_distance(codeword, child.codeword)

            metric_increment = \
                    d * np.log2(p) + (n - d) * np.log(1 - p) \
                    + (1 - B) * n

            child.metric = node.metric + metric_increment

            self.nodes.put(child)

    def decode_node(self, received_sequence):
        """Returns the node corresponding to the decoded path."""
        # Run until we reach the first full-length path
        while True:
            node = self.nodes.get()
            if node.depth == len(received_sequence):
                return node
            self.extend(node, received_sequence[node.depth])

    def decode(self, received_sequence):
        """Returns the decoded bit sequence."""
        return self.decode_node(received_sequence).input_history()

    def E0(self, rho):
        # Compared with (3b) of the tree code paper, 1 = log 2 and the summation
        # have been simplified away
        p = self.p
        return rho - (1 + rho) * np.log2(p**(1/(1+rho)) + (1 - p)**(1/(1+rho)))

    class Node(Node):
        """A node with a comparison operator for use in a min-priority queue."""
        def __lt__(self, other):
            return self.metric > other.metric
