import matplotlib.pyplot as plt
import numpy as np
from joint.coding import SpiralMap

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

def plot_lloyd_max_tracker(distr, enc, dec, tracker, x_hit=None):
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
    plt.plot(tracker.x, tracker.fx, color='orange')
    plt.plot(tracker.w_x, tracker.w_fx, color='purple')
    # plt.xlim(-20, 20)
    # plt.ylim(-0.05, 0.4)
    plt.axis('tight')


def plot_spiral(spiral_map):
    s = np.linspace(0, 7, num=1000)

    # Positive s
    x, y = list(zip(*map(spiral_map.encode, s)))
    plt.plot(x, y, 'orange')

    # Negative s
    x, y = list(zip(*map(spiral_map.encode, -s)))
    plt.plot(x, y, 'lightblue')

    plt.axis('square')
    plt.axis([-22, 22, -22, 22])

def plot_spiral_decode():
    spiral_map = SpiralMap(2, 3)
    fig = plt.figure()
    plt.title("Closest point on Archimedean bi-spiral")
    plot_spiral(spiral_map)

    while True:
        # Retrieve a point that the user clicks
        points = []
        while not points:
            points = plt.ginput(1)
        received = points[0]

        s = spiral_map.decode(received)
        decoded = spiral_map.encode(s)

        plt.scatter([received[0]], [received[1]], color='tomato')
        plt.plot([received[0], decoded[0]], [received[1], decoded[1]],
                color='tomato')
        plt.scatter([decoded[0]], [decoded[1]], color='tomato')

        fig.canvas.draw()

