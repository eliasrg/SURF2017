from itertools import islice
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.integrate import quad

from simulation import Simulation, Parameters
import separate.coding.source.lloyd_max as lm
from separate.coding.channel.convolutional import ConvolutionalCode, Node, \
        hamming_distance, ViterbiDecoder, StackDecoder


n_runs = 1 << 0
T = 1 << 5

LQG_average_trajectories = []
def simulate(plots=False):
    global params, sim, LQG_average_trajectories
    for SNR in [2]: # SNR irrelevant for an IntegerChannel
        print("SNR = {}".format(SNR))
        params = Parameters(
                T = T,
                alpha = 2,
                W = 1, V = 0, # Lloyd-Max paper assumes no observation noise
                Q = 1, R = 1, F = 1)
        params.setRates(KC = 1, KS = 1)
        # params.setAnalog(SNR)
        # params.setScheme('joint')
        params.setDigital(n_codewords = 2)
        params.setScheme('lloyd-max')

        LQG_trajectories = []
        for i in range(n_runs):
            sim = Simulation(params)
            # LQG_trajectory = tuple(sim.LQG.evaluate(t)
            #                        for t in sim.simulate(T))
            LQG_trajectory = []
            if plots and params.scheme == 'lloyd-max':
                plot_lloyd_max(sim.mutual_state.distr,
                        sim.mutual_state.lm_encoder,
                        sim.mutual_state.lm_decoder, x_hit=sim.plant.x)
            try:
                for t in sim.simulate(T):
                    print("Run {:d}, t = {:d}".format(i, t))
                    LQG_trajectory.append(sim.LQG.evaluate(t))
                    if plots and params.scheme == 'lloyd-max':
                        plot_lloyd_max_ms(sim.mutual_state.distr,
                                sim.mutual_state.lm_encoder,
                                sim.mutual_state.lm_decoder, sim.mutual_state,
                                x_hit=sim.plant.x)
            except ValueError:
                print("Divide by zero!")
            except KeyboardInterrupt:
                print("Keyboard interrupt!")
            LQG_trajectories.append(LQG_trajectory)

        LQG_slices = list(zip(*LQG_trajectories))
        LQG_average_trajectory = np.array(list(map(np.mean, LQG_slices)))
        LQG_average_trajectories.append(LQG_average_trajectory)
        print("  Average power over channel: {:.4f}".format(
            sim.channel.average_power()))

    globals().update(params.all()) # Bring parameters into scope

def plot():
    plot = plt.figure()

    for LQG_average_trajectory in LQG_average_trajectories:
        # Plot in dB
        plt.xlabel("Time [steps]")
        plt.ylabel("Average LQR [dB]")
        plt.figure(plot.number)
        plt.plot(list(islice(range(0, T + 1), len(LQG_average_trajectory))),
            10 * np.log10(LQG_average_trajectory))

    if params.analog:
        plt.plot((1, T),
            10 * np.log10(params.LQR_inf_lower_bound()) * np.ones(2), 'r--')

    plt.ylim(0, 50)
    plt.grid()


def plot_lloyd_max(distr, enc, dec, x_hit=None):
    plt.figure()
    plt.scatter(dec.levels, np.zeros(len(dec.levels)), color='red')
    # plt.scatter(enc.boundaries, np.zeros(len(enc.boundaries)),
    #         color='purple', s=3)
    for boundary in enc.boundaries:
        plt.plot([boundary, boundary], [-0.01, distr.pdf(boundary)], color='gray')
    # plt.scatter([distr.mean()], [distr.pdf(distr.mean())], color='green')
    if x_hit is not None: plt.scatter([x_hit], [0], marker='x')
    a = max(distr.a, -20)
    b = min(distr.b, 20)
    x = np.linspace(a, b, num=10000)
    plt.plot(x, distr.pdf(x))
    # plt.xlim(-20, 20)
    # plt.ylim(-0.05, 0.4)
    plt.axis('tight')

def plot_lloyd_max_ms(distr, enc, dec, ms, x_hit=None):
    plt.figure()
    plt.scatter(dec.levels, np.zeros(len(dec.levels)), color='red')
    # plt.scatter(enc.boundaries, np.zeros(len(enc.boundaries)),
    #         color='purple', s=3)
    for boundary in enc.boundaries:
        plt.plot([boundary, boundary], [-0.01, distr.pdf(boundary)], color='gray')
    # plt.scatter([distr.mean()], [distr.pdf(distr.mean())], color='green')
    if x_hit is not None: plt.scatter([x_hit], [0], marker='x')
    a = max(distr.a, -20)
    b = min(distr.b, 20)
    x = np.linspace(a, b, num=10000)
    plt.plot(x, distr.pdf(x))
    plt.plot(ms.x, ms.fx, color='orange')
    plt.plot(ms.w_x, ms.w_fx, color='purple')
    # plt.xlim(-20, 20)
    # plt.ylim(-0.05, 0.4)
    plt.axis('tight')

def generate_plot_lloyd_max(n_levels):
    distr = st.norm
    enc, dec = lm.generate(n_levels, distr)
    plot_lloyd_max(distr, enc, dec)

def test_update(i=4):
    global ms
    from separate.coding.source import MutualState
    ms = MutualState(sim, 10)
    plot_lloyd_max(ms.distr, ms.lm_encoder, ms.lm_decoder)
    ms.update(i, debug_globals=globals())
    plot_lloyd_max(ms.distr, ms.lm_encoder, ms.lm_decoder)

def convolutional_test():
    """Test of convolutional encoder (from a homework problem)."""
    Gs = [
        np.array([[1, 1, 1]]).transpose(),
        np.array([[0, 0, 1]]).transpose(),
        np.array([[0, 1, 0]]).transpose(),
        np.array([[0, 1, 0]]).transpose(),
        np.array([[0, 0, 1]]).transpose(),
        np.array([[0, 1, 1]]).transpose()]
    cc = ConvolutionalCode(3, 1, Gs)
    msg = [np.array([[x]]) for x in [1,1,1,0,1,0,0,1,0,0,0,0,0]]
    code = cc.encode_sequence(msg)
    nominal_code = np.array([1,1,1, 1,1,0, 1,0,0, 0,0,1, 1,1,0, 0,0,1, 0,0,0,
                             1,1,0, 0,0,0, 0,0,1, 0,1,0, 0,0,1, 0,1,1])
    assert (np.array(code).flatten() == nominal_code).all()

    return cc, msg, nominal_code

def viterbi_test():
    """Test of the Viterbi algorithm (from a homework problem)."""
    Gs = [
        np.array([[1, 1]]).transpose(),
        np.array([[0, 1]]).transpose(),
        np.array([[1, 1]]).transpose()]
    code = ConvolutionalCode(2, 1, Gs)
    received_sequence = [np.array([x]).transpose() for x in
            [[1,0], [1,1], [1,0], [1,1], [1,0], [1,1], [1,0]]]
    decoder = ViterbiDecoder(code)
    decoder.decode(received_sequence)
    nominal_input = [np.array([[x]]) for x in [0, 1, 1, 1, 0, 0, 0]]

    # In the homework problem, it was known that the last two bits were 0.
    decoded_inputs = [history for history in map(list, decoder.best_inputs)
            if history[-2:] == [np.array([[0]])] * 2]

    assert all(node.depth == len(nominal_input) for node in decoder.best_nodes)
    assert decoder.best_hamming_distance == 3
    assert decoded_inputs == [nominal_input]

    return code, received_sequence, decoded_inputs[0]

def stack_test():
    """Test of the Stack algorithm (from Viterbi and Omura, Figure 6.1)."""
    Gs = [
        np.array([[1, 1]]).transpose(),
        np.array([[1, 0]]).transpose(),
        np.array([[1, 1]]).transpose()]
    code = ConvolutionalCode(2, 1, Gs)
    received_sequence = [np.array([x]).transpose() for x in
            [[0,1], [1,0], [0,1], [1,0], [1,1]]]
    decoder = StackDecoder(code, 0.03)
    decoded_input = list(decoder.decode(received_sequence))
    nominal_input = [np.array([[x]]) for x in [1, 0, 1, 0, 0]]

    assert decoded_input == nominal_input

    return code, received_sequence, decoded_input


def show(delay=0):
    if delay != 0:
        from time import sleep
        sleep(delay)
    plt.show(block=False)
