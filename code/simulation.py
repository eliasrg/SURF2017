def simulate(plant, channel):
    yield
    while True:
        b = channel.transmit(plant.a)
        u = -b * plant.y
        plant.step(u)
        yield
