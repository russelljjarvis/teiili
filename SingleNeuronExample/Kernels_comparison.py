#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat 25 Mar 17:45:26 2017

@author: Karla Burelo
"""
#==============================================================================
#This code has four different population of hidden neurons, one poisson group  
#or spike generator of neurons that provides input spikes to each of the four
#groups. Each neuron receives input from one neuorn. Here we want to visualize 
#the different effects of an input in the EPSP using different kernels defined 
#in Tapson et al.(2013) 
#==============================================================================
#start_scope()

from brian2 import *
import numpy as np

from synapseEquations import *
from neuronEquations import Silicon
from tools import *
from synapseParams import *
from neuronParams import *

#==============================================================================
# Parameters & Equations for Neuron Groups
#==============================================================================
eqSil = Silicon()

#InputNeurons = PoissonGroup(1, np.arange(1)*Hz + 10*Hz)
indices = array([0,0,0])
times = array([100,150,300])*ms
InputNeurons = SpikeGeneratorGroup(1, indices, times)

#Output Neurons
G_1 = NeuronGroup(1, method = 'euler', refractory = 0.5 * ms, **eqSil)
G_2 = NeuronGroup(1, method = 'euler', refractory = 0.5 * ms, **eqSil)
G_3 = NeuronGroup(1, method = 'euler', refractory = 0.5 * ms, **eqSil)
G_4 = NeuronGroup(1, method = 'euler', refractory = 0.5 * ms, **eqSil)

setParams(G_1,SiliconNeuronP, debug = False)
setParams(G_2,SiliconNeuronP, debug = False)
setParams(G_3,SiliconNeuronP, debug = False)
setParams(G_4,SiliconNeuronP, debug = False)

#==============================================================================
# Parameters & Equations for Synapses Groups
#==============================================================================
sdict_1, argument_1 = KernelsSynapses(kernel = 'alpha', debug = True)
sdict_2, argument_2 = KernelsSynapses(kernel = 'resonant', debug = True)
sdict_3, argument_3 = KernelsSynapses(kernel = 'expdecay', debug = True)
sdict_4, argument_4 = KernelsSynapses(kernel = 'gaussian', debug = True)

S_1 = Synapses(InputNeurons, G_1, method = 'euler', **sdict_1)
S_2 = Synapses(InputNeurons, G_2, method = 'euler', **sdict_2)
S_3 = Synapses(InputNeurons, G_3, method = 'euler', **sdict_3)
S_4 = Synapses(InputNeurons, G_4, method = 'euler', **sdict_4)

S_1.connect(True)
S_2.connect(True)
S_3.connect(True)
S_4.connect(True)

S_1.w = 100*pA
S_2.w = 100*pA
S_3.w = 100*pA
S_4.w = 2*pA

S_1.t_spike = 1 * second
S_2.t_spike = 1 * second
S_3.t_spike = 1 * second
S_4.t_spike = 1 * second

setParams(S_1 , KernelP, debug = False)
setParams(S_2 , KernelP, debug = False)
setParams(S_3 , KernelP, debug = False)
setParams(S_4 , KernelP, debug = False)

#==============================================================================
# Monitoring results
#==============================================================================
M_1 = StateMonitor(G_1,['Imem'], record = True)
M_S1 = StateMonitor(G_1,[ 'Iin_ex'], record = True)
M_S2 = StateMonitor(G_2, ['Iin_ex'], record = True)
M_S3= StateMonitor(G_3,[ 'Iin_ex'], record = True)
M_S4= StateMonitor(G_4,[ 'Iin_ex'], record = True)
spikemon_poisson = SpikeMonitor(InputNeurons)

run(800*ms)

#==============================================================================
# Figures
#==============================================================================
cInd=0
neuron1=spikemon_poisson.i*(spikemon_poisson.i==cInd)
timestamp=[]

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
ax1.plot(M_S1.t/ms, M_S1.Iin_ex[0]/pA,'-g')
ax1.set_title('Input excitatory current to the neuron (pA)')
ax2.plot(M_S2.t/ms, M_S2.Iin_ex[0]/pA,'-g')
ax3.plot(M_S3.t/ms, M_S3.Iin_ex[0]/pA,'-g')
ax4.plot(M_S4.t/ms, M_S4.Iin_ex[0]/pA,'-g')
ax1.set_ylabel('alpha')
ax2.set_ylabel('resonant')
ax3.set_ylabel('expdecay')
ax4.set_ylabel('gaussian')
for i in range(neuron1.shape[0]):
    if neuron1[i]==cInd:
        timestamp.append(spikemon_poisson.t[i])
for t in timestamp:
    ax1.axvline(t/ms, ls='--', c='r', lw=1)
    ax2.axvline(t/ms, ls='--', c='r', lw=1)
    ax3.axvline(t/ms, ls='--', c='r', lw=1)
    ax4.axvline(t/ms, ls='--', c='r', lw=1)
xlabel('Time (ms)')
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0.5)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

show()

