#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:43:49 2018

This is just an example, how poisson neurons with an inhomogenous poisson firing rate based on synaptic input or other things
could be implemented using Brian2.

@author: alpha
"""

import time
import os
import pandas as pd
import numpy as np
from brian2 import device, codegen, defaultclock, NeuronGroup, Synapses, run,\
                   SpikeMonitor, StateMonitor, start_scope, Network,\
                   implementation, declare_types, check_units, prefs,TimedArray,\
                   PoissonGroup
from brian2 import mV, mA, ms, ohm, uA, pA, Hz



poissongroup = PoissonGroup(1,rates=100*Hz)

testneuron = NeuronGroup(1,'dV/dt=1/(10*ms) : 1', threshold='V>10', reset='V=0')


testsynapse = Synapses(testneuron,poissongroup,'',on_pre='rates_post+=10*Hz')

testsynapse.connect(True)

run(205*ms)

print(testneuron.V)
print(poissongroup.rates)
