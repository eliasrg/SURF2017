class TrivialEncoder:
    """Sends a message without encoding it over the channel."""
    def __init__(self, sim):
        self.sim = sim

    def encode(self, *msg):
        return msg

class TrivialDecoder:
    """Decodes a trivially encoded message (i.e. returns it unchanged)."""
    def __init__(self, sim):
        self.sim = sim

    def decode(self, *code):
        return code
