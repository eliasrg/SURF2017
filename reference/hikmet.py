import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve
import matplotlib.pyplot as plt
from tqdm import trange


import sys, os
sys.path.insert(0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'code')))

from itertools import islice
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.integrate import quad

from simulation import Simulation, Parameters
from measurements import Measurement
from plotting import plot_lloyd_max, plot_lloyd_max_tracker, \
        plot_spiral, plot_spiral_decode


# Constants
RESOLUTION=1<<7

class Distribution:
    def __init__(self, interval, pdf):
        self.interval=interval
        self.pdf=pdf
    @classmethod
    def bySamples(cls, x, fx): # Interpolate to get the pdf
        # Use logarithmic interpolation to preserve log-concavity
        dx=x[1]-x[0]
        fx=np.array(fx, dtype = float) / sum(fx) / dx
        Fx=np.cumsum(fx)*dx
        v1 = sum(1 for i in Fx if i < 1e-5)
        v2 = sum(1 for i in Fx if i < 1-1e-5)
        x=x[v1:v2]
        fx=fx[v1:v2]
        fx=np.array(fx, dtype = float) / sum(fx) / dx
        logfx=np.log(fx)
        logpdf=interp1d(x, logfx, kind='linear',
                        bounds_error=False, fill_value=float('-inf'))
        pdf = lambda t : np.exp(logpdf(t))
        return cls((x[0],x[-1]), pdf)
    def convolution(d1, d2):
        a1,b1 = d1.interval
        a2,b2 = d2.interval
        delta = max(b1-a1,b2-a2) / float(RESOLUTION)
        f1=[d1.pdf(i) for i in np.arange(a1,b1,delta)]
        f2=[d2.pdf(i) for i in np.arange(a2,b2,delta)]
        fx=convolve(f1, f2)
        x=[a1+a2+delta*i for i in range(len(fx))]
        return Distribution.bySamples(x, fx)

def LM(distribution, n):
    # Some definitions
    maxiter=1<<10
    N=RESOLUTION
    a,b = distribution.interval
    x=np.linspace(a,b,N)
    fx=np.array([distribution.pdf(i) for i in x])
    fx[np.isnan(fx)]=0
    dx=(b-a) / (N-1.)
    Fx=np.cumsum(fx)*dx
    index=lambda y: int(min(N-1, max(0, np.round((y-a) / float(dx)))))

    # Initialization
    c=np.zeros(n)
    # p=np.array([x[i] for i in sorted(np.random.randint(0, N, n-1))])
    p=np.array([x[int(i)] for i in np.round(np.linspace(0, N, num=n+1)[1:-1])])
    # Loop
    error=1
    iteration=0
    while error > 0 and iteration<maxiter:
        iteration +=1
        # centers from boundaries
        pin=[0]+[index(i) for i in p]+[N-1]
        for i in range(n):
            c[i]=sum(x[j]*fx[j] for j in range(pin[i],pin[i+1]+1))\
                /sum(     fx[j] for j in range(pin[i],pin[i+1]+1))
        pin_temp=pin
        # boundaries from centers
        p=(c[:-1]+c[1:]) / 2.
        pin=[0]+[index(i) for i in p] + [N-1]
        error=sum(abs(pin_temp[i]-pin[i]) for i in range(n+1))
    return ([a]+list(p)+[b],c)



def plot_lloyd_max(distr, boundaries, levels, x_hit=None):
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

def plot_lloyd_max_tracker(distr, boundaries, levels, d1, fw, x_hit=None):
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

def show(delay=0):
    if delay != 0:
        from time import sleep
        sleep(delay)
    plt.show(block=False)


# m = Measurement.load('./data/separate/varying-SNR/noiseless--4.p')
# m.w[0] = m.x[0]
# w_sequence = m.get_noise_record().w_sequence

# Parameters
A = 1.5
W = 1.0 # wt ~ iid N(0,W)
T = 1<<8
# A = m.params.alpha
# W = m.params.W
# T = 1 << 7

# Definitions
fw = Distribution((-10,10),
        lambda x : W * np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)
        )# pdf of w_t: N(0,W) with support (-10,10)

num_iter = 1<<9
avg=np.zeros(T)
LQG_avg = np.zeros(T)
for it in trange(num_iter):
    # Initialization
    l = 0
    x = 0
    u = 0
    prior_dist = fw # prior pdf of x

    e=[]
    x2s = []
    # Loop
    for t in range(T):
        w = W * np.random.randn() # should be W
        # w = w_sequence.pop(0)
        x = A * x + u + w

        (p,c) = LM(prior_dist, 2)

        # if t == 0:
        #     plot_lloyd_max(prior_dist, p, c, x_hit=x)
        # else:
        #     plot_lloyd_max_tracker(prior_dist, p, c, d1, fw, x_hit=x)

        l = sum(1 for i in p[1:-1] if i <= x) # encode
        x_hat = c[l] # decode

        u = - A * x_hat

        d1 = Distribution((A*p[l]+u,A*p[l+1]+u), lambda x: prior_dist.pdf((x-u) / float(A)))
        prior_dist = Distribution.convolution(d1, fw)

        e+=[(x-x_hat)**2]
        x2s += [x**2]
    error=np.cumsum(e)/np.arange(T)
    avg+=error/num_iter
    LQG=np.cumsum(x2s)/np.arange(T)
    LQG_avg += LQG/num_iter

LQGcosts = A**2 * avg + W
