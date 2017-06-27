def simulate(plant, channel, encoder, decoder, LQG, T):
    t = 1
    yield t
    while t < T:
        x_est = decoder.decode(t,
                *(channel.transmit(p) for p in encoder.encode(t, plant.y)))
        u = -plant.alpha * x_est
        plant.step(u)
        LQG.step(u)
        t += 1
        yield t
