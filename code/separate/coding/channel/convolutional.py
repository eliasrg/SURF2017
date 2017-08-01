from utilities import hamming_distance

import itertools as it
import numpy as np
from queue import PriorityQueue
from numbers import Number

class ConvolutionalCode:
    def __init__(self, n, k, Gs):
        self.n = n # Output block length
        self.k = k # Input block length
        self.Gs = Gs # Generator matrices G_1, ..., G_T ∊ ℤ2^(n×k)
        assert all(G.shape == (n,k) for G in Gs)

    def encode(self, data):
        """Performs one encoding step.
    data is a sequence of column vectors in ℤ2^k, in chronological order."""
        return self.encode_reversed(reversed(list(data)))

    def encode_reversed(self, reversed_data):
        """Performs one encoding step.
    data is a sequence of column vectors in ℤ2^k,
    in reverse chronological order."""
        return sum(G @ b for G, b in zip(self.Gs, reversed_data)) % 2

    def encode_sequence(self, data):
        """Encodes an entire sequence.
    data is a list of column vectors in ℤ2^k."""
        data = list(data)
        return [self.encode(data[:i+1]) for i in range(len(data))]

    def rate(self):
        return self.k / self.n


class Node:
    """A class of nodes for use in decoding algorithms.
    The root of the tree is a node Node(code). The children of a node are
    created by node.extend().
    Instance variables:
    ∙ ConvolutionalCode code
    ∙ Node parent (or None)
    ∙ depth ∊ ℕ
    and for all nodes except the root:
    ∙ input_block ∊ ℤ2^k
    ∙ codeword ∊ ℤ2^n"""

    def __init__(self, code, parent=None, input_block=None):
        self.code = code
        self.parent = parent
        self.depth = 0 if self.is_root() else self.parent.depth + 1

        if input_block is not None:
            assert not self.is_root()
            self.input_block = input_block
            # codeword is set by the call to parent.extend()

    def is_root(self):
        return self.parent is None

    def reversed_input_history(self):
        node = self
        while not node.is_root():
            yield node.input_block
            node = node.parent

    def input_history(self):
        return reversed(list(self.reversed_input_history()))

    def extend(self, possible_input_blocks=None):
        """Creates and yields the children of this node.
        if possible_input_blocks is None (the default), assumes that all binary
        input vectors of length k are possible."""

        # The next codeword is (if the next time is t)
        # ct = Gt b1 + G(t-1) b2 + ... + G1 bt.
        # Compute everything but the last term (i.e. the above with bt = 0)
        # and call it c.
        zero = np.zeros([self.code.k, 1], int)
        c = self.code.encode_reversed(
                it.chain([zero], self.reversed_input_history()))

        # Iterate over all possible input blocks
        if possible_input_blocks is None:
            possible_input_blocks = it.product([0,1], repeat=self.code.k)
        for bits in possible_input_blocks:
            input_block = np.array([bits]).transpose()

            # Create a new node
            node = self.__class__(self.code, self, input_block)

            # Calculate the expected output at the new node
            node.codeword = c ^ (self.code.Gs[0] @ input_block % 2)

            yield node


class ViterbiDecoder:
    """The Viterbi algorithm for maximum-likelihood decoding of a tree code
    (i.e. convolutional code with infinite shift register). For such a code,
    the algorithm simplifies to a naive exhaustive search with no dynamic
    programming."""

    def __init__(self, code):
        self.code = code

    def decode(self, received_sequence):
        # Annotate each node with the Hamming distance (dist) between the
        # received and predicted code sequences. No extra fields are needed for
        # backtracking because every node has only one parent
        root = Node(self.code)
        root.dist = 0
        current_layer = [root]
        for codeword in received_sequence:
            next_layer = []
            for node in current_layer:
                children = list(node.extend())
                for child in children:
                    child.dist = node.dist + \
                            hamming_distance(codeword, child.codeword)
                next_layer += children
            current_layer = next_layer

        best_node = min(current_layer, key=lambda node: node.dist)

        # Save additional results as fields of the object
        self.best_hamming_distance = best_node.dist
        self.final_layer = current_layer
        self.best_nodes = [node for node in self.final_layer
                if node.dist == self.best_hamming_distance]
        self.best_inputs = [node.input_history() for node in self.best_nodes]

        return best_node.input_history()


class StackDecoder:
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

    def E0(self, rho):
        # Compared with (3b) of the tree code paper, 1 = log 2 and the summation
        # have been simplified away
        p = self.p
        return rho - (1 + rho) * np.log2(p**(1/(1+rho)) + (1 - p)**(1/(1+rho)))

    def decode(self, received_sequence):
        nodes = PriorityQueue()
        root = StackDecoder.Node(self.code)
        root.metric = 0
        nodes.put(root)

        # Run until we reach the first full-length path
        while True:
            node = nodes.get()
            if node.depth == len(received_sequence):
                return node.input_history()

            for child in node.extend():
                # Calculate the metric increment
                p = self.p
                k = self.code.k
                n = self.code.n
                B = self.bias
                d = hamming_distance(
                        received_sequence[child.depth - 1], child.codeword)

                metric_increment = \
                        d * np.log2(p) + (n - d) * np.log(1 - p) \
                        + (1 - B) * n

                child.metric = node.metric + metric_increment

                nodes.put(child)

    class Node(Node):
        """A node with a comparison operator for use in a min-priority queue."""
        def __lt__(self, other):
            return self.metric > other.metric
