def simulate(plant, channel):
    yield
    while True:
        b = channel.transmit(plant.y)
        u = -plant.a * b
        plant.step(u)
        yield
