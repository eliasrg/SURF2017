import numpy as np
import itertools as it

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

    def reversed_input_history(self, stop_at=None):
        """The reversed input history back to, but not including, the node
        stop_at. If stop_at is None (the default), return the entire input
        history."""
        node = self
        while not node.parent == stop_at:
            yield node.input_block
            node = node.parent

    def input_history(self, stop_at=None):
        """The reversed input history back to, but not including, the node
        stop_at. If stop_at is None (the default), return the entire input
        history."""
        return reversed(list(self.reversed_input_history(stop_at)))

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

    def first_common_ancestor(a, b):
        # Let a be the deepest node
        if a.depth < b.depth:
            a, b = b, a

        # Make sure a and b are at the same depth
        while a.depth > b.depth:
            a = a.parent

        # Step upwards until a common ancestor is found
        while a is not b:
            a = a.parent
            b = b.parent

        return a
