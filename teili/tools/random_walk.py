#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:55:59 2018

@author: alpha
"""

import numpy as np
from brian2 import ms, us

# import random
# import itertools
from numpy.fft import rfft, irfft


def normalize(y, x=None):
    """
    from https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/generator.py
    normalize power in y to a (standard normal) white noise signal.
    Optionally normalize to power in signal `x`.
    #The mean power of a Gaussian with :math:`\\mu=0` and :math:`\\sigma=1` is 1.
    """
    # return y * np.sqrt( (np.abs(x)**2.0).mean() / (np.abs(y)**2.0).mean() )
    if x is not None:
        x = (np.abs(x) ** 2.0).mean()
    else:
        x = 1.0
    return y * np.sqrt(x / (np.abs(y) ** 2.0).mean())


def pink(N, state=None):
    """
    from https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/generator.py
    Generates pink noise.
    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`
    Pink noise has equal power in bands that are proportionally wide.
    Power density decreases with 3 dB per octave.
    """
    # This method uses the filter with the following coefficients.
    # b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    # a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
    # return lfilter(B, A, np.random.randn(N))
    # Another way would be using the FFT
    # x = np.random.randn(N)
    # X = rfft(x) / N
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)  # +1 to avoid divide by zero
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)
