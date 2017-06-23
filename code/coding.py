class TrivialEncoder:
    def encode(self, y):
        return (y,)

class TrivialDecoder:
    def decode(self, *code):
        assert(len(code) == 1)
        return code[0]
