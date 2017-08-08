from .node import Node
from utilities import hamming_distance

class NaiveMLDecoder:
    """A naive exhaustive search for maximum-likelihood decoding of a
    convolutional code. For a tree code (i.e. a convolutional code with infinite
    shift register), the Viterbi algorithm simplifies to precisely this."""

    def __init__(self, code):
        self.code = code

    def decode(self, received_sequence, save=False):
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

        if save:
            # Save additional results as fields of the object
            self.best_hamming_distance = best_node.dist
            self.final_layer = current_layer
            self.best_nodes = [node for node in self.final_layer
                    if node.dist == self.best_hamming_distance]
            self.best_inputs = [node.input_history() for node in self.best_nodes]

        return best_node.input_history()
