class TrivialEncoder:
    def __init__(self, sim):
        self.sim = sim

    def encode(self, *msg):
        return msg

class TrivialDecoder:
    def __init__(self, sim):
        self.sim = sim

    def decode(self, *code):
        return code
