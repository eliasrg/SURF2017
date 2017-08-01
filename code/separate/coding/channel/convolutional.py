import itertools as it
import numpy as np

class ConvolutionalCode:
    def __init__(self, n, k, Gs):
        self.n = n # Output block length
        self.k = k # Input block length
        self.Gs = Gs # Generator matrices G_1, ..., G_T ∊ ℤ2^(n×k)
        assert(all(G.shape == (n,k) for G in Gs))

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


class Node:
    """A class of nodes for use in decoding algorithms.
    The root of the tree is a node Node(code). The children of a node are
    created by node.extend().
    Instance variables:
    ∙ ConvolutionalCode code
    ∙ Node parent (or None)
    and for all nodes except the root:
    ∙ input_block ∊ ℤ2^k
    ∙ codeword ∊ ℤ2^n"""

    def __init__(self, code, parent=None, input_block=None):
        self.code = code
        self.parent = parent
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
            node = Node(self.code, self, input_block)

            # Calculate the expected output at the new node
            node.codeword = c ^ (self.code.Gs[0] @ input_block % 2)

            yield node


def hamming_distance(a, b):
    """Computes the Hamming distance between two binary column vectors."""
    return np.abs(a - b).sum()
