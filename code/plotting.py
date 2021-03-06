# Copyright (c) 2017 Elias Riedel Gårding
# Licensed under the MIT License

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

def plot_lloyd_max_hikmet(distr, boundaries, levels, x_hit=None):
    plt.figure()
    plt.scatter(levels, np.zeros(len(levels)), color='red')
    # plt.scatter(boundaries, np.zeros(len(boundaries)),
    #         color='purple', s=3)
    for boundary in boundaries:
        plt.plot([boundary, boundary], [-0.01, distr.pdf(boundary)], color='gray')
    # plt.scatter([distr.mean()], [distr.pdf(distr.mean())], color='green')
    if x_hit is not None: plt.scatter([x_hit], [0], marker='x')
    a = max(distr.interval[0], -20)
    b = min(distr.interval[1], 20)
    x = np.linspace(a, b, num=10000)
    plt.plot(x, distr.pdf(x))
    # plt.xlim(-20, 20)
    # plt.ylim(-0.05, 0.4)
    plt.axis('tight')

def plot_lloyd_max_tracker_hikmet(distr, boundaries, levels, d1, fw, x_hit=None):
    plt.figure()
    plt.scatter(levels, np.zeros(len(levels)), color='red')
    # plt.scatter(boundaries, np.zeros(len(boundaries)),
    #         color='purple', s=3)
    for boundary in boundaries:
        plt.plot([boundary, boundary], [-0.01, distr.pdf(boundary)], color='gray')
    # plt.scatter([distr.mean()], [distr.pdf(distr.mean())], color='green')
    if x_hit is not None: plt.scatter([x_hit], [0], marker='x')
    a = max(distr.interval[0], -20)
    b = min(distr.interval[1], 20)
    x = np.linspace(a, b, num=10000)
    plt.plot(x, distr.pdf(x))
    plt.plot(x, (d1.interval[0] <= x) * (x <= d1.interval[1]) * d1.pdf(x), color='orange')
    plt.plot(x, fw.pdf(x), color='purple')
    # plt.xlim(-20, 20)
    # plt.ylim(-0.05, 0.4)
    plt.axis('tight')


def plot_spiral(spiral_map):
    s = np.linspace(0, 7, num=1000)

    plt.rcParams['lines.linewidth'] = 5

    plt.plot(spiral_map.c * s, spiral_map.c * s, '--', color='orange')
    plt.plot(-spiral_map.c * s, -spiral_map.c * s, '--', color='lightblue')

    # Positive s
    x, y = list(zip(*map(spiral_map.encode, s)))
    plt.plot(x, y, 'orange')

    # Negative s
    x, y = list(zip(*map(spiral_map.encode, -s)))
    plt.plot(x, y, 'lightblue')

    plt.axis('square')
    plt.axis([-22, 22, -22, 22])

    plt.xlabel("First channel use ($a_{i_t}$)", fontsize=25)
    plt.ylabel("Second channel use ($a_{i_t + 1}$)", fontsize=25)

def plot_spiral_decode(spiral_map=SpiralMap(2, 3)):
    fig = plt.figure()
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

