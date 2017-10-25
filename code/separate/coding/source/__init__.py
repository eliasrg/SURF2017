# Copyright (c) 2017 Elias Riedel Gårding
# Licensed under the MIT License

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve
import matplotlib.pyplot as plt

from . import lloyd_max

class Encoder:
    def __init__(self, sim, tracker):
        self.sim = sim
        self.tracker = tracker

    def encode(self, *msg):
        # Just encode it with Lloyd-Max
        # (pass it to channel encoder or digital channel)
        i = self.tracker.lm_encoder.encode(*msg)
        self.tracker.update(i)
        return (i,)

    def get_tracker(self):
        return self.tracker


class Decoder:
    def __init__(self, sim, tracker):
        self.sim = sim
        self.tracker = tracker

    def clone(self):
        return self.__class__(self.sim, self.tracker.clone())

    def decode(self, *msg):
        # Decode with the Lloyd-Max decoder
        # (receive from channel encoder or digital channel)
        x_est = self.tracker.lm_decoder.decode(*msg)

        # Update the distribution tracker
        assert len(msg) == 1 # One integer
        i = msg[0]
        self.tracker.update(i)

        return (x_est,)


# Hikmet's code

# Constants
RESOLUTION=1<<7

class Distribution:
    def __init__(self, interval, pdf):
        self.interval=interval
        self.pdf=pdf
        self.is_hikmet = True
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


class DistributionTracker:
    """Keeps track of the distribution of the plant's state."""
    def __init__(self, sim, n_levels, distr=None,
            lm_encoder=None, lm_decoder=None):
        self.sim = sim
        self.n_levels = n_levels

        if distr is None:
            assert lm_encoder is None and lm_decoder is None
            W = self.sim.params.W
            self.fw = Distribution((-10,10),
                    lambda x : W * np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)
                    )# pdf of w_t: N(0,W) with support (-10,10)
            self.distr = self.fw
            boundaries, levels = LM(self.distr,
                    2**self.sim.params.quantizer_bits)
            self.lm_encoder = lloyd_max.Encoder(boundaries)
            self.lm_decoder = lloyd_max.Decoder(levels, boundaries)
        else:
            assert lm_encoder is not None and lm_decoder is not None
            self.distr = distr
            self.lm_encoder = lm_encoder
            self.lm_decoder = lm_decoder

        # DEBUG
        self.distrs = []

    def clone(self):
        new = self.__class__(self.sim, self.n_levels, self.distr,
                self.lm_encoder, self.lm_decoder)

        # DEBUG
        new.distrs = self.distrs[:]
        if hasattr(self, 'x'): new.x = self.x
        if hasattr(self, 'fx'): new.fx = self.fx
        if hasattr(self, 'w_x'): new.w_x = self.w_x
        if hasattr(self, 'w_fx'): new.w_fx = self.w_fx
        if hasattr(self, 'd1'): new.d1 = self.d1
        if hasattr(self, 'fw'): new.fw = self.fw

        return new

    def update(self, i, debug_globals=dict()):
        A = self.sim.params.alpha
        L = self.sim.params.L(self.sim.t)

        x_hat = self.lm_decoder.decode(i)
        u = -L * x_hat

        lo, hi = self.lm_decoder.get_interval(i)
        lo = max(lo, self.distr.interval[0])
        hi = min(hi, self.distr.interval[1])

        self.d1 = Distribution((A*lo+u,A*hi+u), lambda x: self.distr.pdf((x-u) / float(A)))
        self.distr = Distribution.convolution(self.d1, self.fw)

        self.distrs.append(self.distr) # DEBUG

        # DEBUG: For inspecting the local variables interactively
        debug_globals.update(locals())

        boundaries, levels = LM(self.distr, 2**self.sim.params.quantizer_bits)
        self.lm_encoder = lloyd_max.Encoder(boundaries)
        self.lm_decoder = lloyd_max.Decoder(levels, boundaries)
