def simulate(plant, channel, encoder, decoder, LQG):
    yield
    while True:
        x_est = decoder.decode(
                *(channel.transmit(p) for p in encoder.encode(plant.y)))
        u = -plant.a * x_est
        plant.step(u)
        LQG.step(u)
        yield
