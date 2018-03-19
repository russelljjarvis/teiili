#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:50:14 2017

This example shows mainly how the standalone code generation with changeable parameters works
It simulates the core of the sequence learning architecture

if you are using anaconda, you might need to update libgcc before the cpp code generation works correctly (conda install libgcc) Version 5.2. works for me.


@author: alpha
"""
import numpy as np
import os
from collections import OrderedDict

from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second, Network, prefs, SpikeGeneratorGroup, NeuronGroup,\
    Synapses, SpikeMonitor, StateMonitor, figure, plot, show, xlabel, ylabel,\
    seed, xlim, ylim, subplot, network_operation, set_device, device, TimedArray,\
    defaultclock, profiling_summary, codegen, size, pamp, pfarad, msecond
#from brian2 import *

from NCSBrian2Lib.building_blocks.sequence_learning import SequenceLearning

from NCSBrian2Lib.models.neuron_models import ExpAdaptIF
from NCSBrian2Lib.models.synapse_models import ReversalSynV

from NCSBrian2Lib.models.parameters.exp_adapt_if_param import parameters as neuron_parameters
from NCSBrian2Lib.models.parameters.exp_syn_param import parameters as syn_parameters

from NCSBrian2Lib.tools.cpptools import buildCppAndReplace, collectStandaloneParams, run_standalone
from NCSBrian2Lib.core.network import NCSNetwork

standaloneDir = os.path.expanduser('~/SL_standalone')
#isStandalone = True

try:
    seqNet.hasRun
    print('Network has been built already, rebuild it...')
    device.reinit()
    device.activate(directory=standaloneDir, build_on_run=False)
except:
    set_device('cpp_standalone', directory=standaloneDir, build_on_run=False)


prefs.devices.cpp_standalone.openmp_threads = 4

defaultclock.dt = 100 * us

duration = 330 * ms
nPerGroup = 8
nElem = 3
nOrdNeurons = nElem * nPerGroup

slParams = {'synInpOrd1e_weight': 1.3,
            'synOrdMem1e_weight': 1.1,
            'synMemOrd1e_weight': 0.16,
            # local
            'synOrdOrd1e_weight': 1.04,
            'synMemMem1e_weight': 1.54,
            # inhibitory
            'synOrdOrd1i_weight': -1.95,
            'synMemOrd1i_weight': -0.384,
            'synCoSOrd1i_weight': -1.14,
            'synResetOrd1i_weight': -1.44,
            'synResetMem1i_weight': -2.6,
            # refractory
            'gOrdGroups_refP': 1.7 * ms,
            'gMemGroups_refP': 2.3 * ms
            }

SequenceLearningExample = SequenceLearning(name='Seq',
                                           neuron_eq_builder=ExpAdaptIF,
                                           synapse_eq_builder=ReversalSynV,
                                           neuronParams=neuron_parameters,
                                           synapseParams=syn_parameters,
                                           blockParams=slParams,
                                           numElements=nElem,
                                           numNeuronsPerGroup=nPerGroup,
                                           debug=False)

#for key in SLGroups: print(key)

# Input to start sequence manually
tsInput = np.concatenate((np.ones((nPerGroup,), dtype=np.int) * 5, np.ones((nPerGroup,), dtype=np.int) * 6,
                          np.ones((nPerGroup,), dtype=np.int) * 7
                          )) * ms
indInput = np.mod(np.arange(size(tsInput)), nPerGroup)
SequenceLearningExample.inputGroup.set_spikes(indices=indInput, times=tsInput)

# CoS group
tsCoS = np.concatenate((np.ones((nPerGroup,), dtype=np.int) * 100, np.ones((nPerGroup,), dtype=np.int) * 101,
                        np.ones((nPerGroup,), dtype=np.int) * 102, np.ones((nPerGroup,), dtype=np.int) * 103,
                        np.ones((nPerGroup,), dtype=np.int) * 104, np.ones((nPerGroup,), dtype=np.int) * 105,

                        np.ones((nPerGroup,), dtype=np.int) * 200, np.ones((nPerGroup,), dtype=np.int) * 201,
                        np.ones((nPerGroup,), dtype=np.int) * 202, np.ones((nPerGroup,), dtype=np.int) * 203,
                        np.ones((nPerGroup,), dtype=np.int) * 204, np.ones((nPerGroup,), dtype=np.int) * 205
                        )) * ms
indCoS = np.mod(np.arange(size(tsCoS)), nPerGroup)
SequenceLearningExample.cosGroup.set_spikes(indices=indCoS, times=tsCoS)

# reset group
tsReset = np.concatenate((np.ones((nPerGroup,), dtype=np.int) * 300, np.ones((nPerGroup,), dtype=np.int) * 301,
                          np.ones((nPerGroup,), dtype=np.int) * 302, np.ones((nPerGroup,), dtype=np.int) * 303,
                          np.ones((nPerGroup,), dtype=np.int) * 304, np.ones((nPerGroup,), dtype=np.int) * 305,
                          np.ones((nPerGroup,), dtype=np.int) * 306, np.ones((nPerGroup,), dtype=np.int) * 307
                          )) * ms
indReset = np.mod(np.arange(size(tsReset)), nPerGroup)
SequenceLearningExample.resetGroup.set_spikes(indices=indReset, times=tsReset)

seqNet = NCSNetwork()
#seqNet = Network()
seqNet.add(SequenceLearningExample,SequenceLearningExample.Monitors)

# this is how you add additional parameters that you want to change in the standalone run (you just have to find out their name...)
# taugIi in this case is valid vor all neurons!
# please note, that this is string replacement, so if you have another state variable that is called e.g. GammataugIi, this would also be replaced!
#seqNet.add_standaloneParams(gOrd_Seq_b=0.0805*nA, taugIi=6*ms)

seqNet.build()

#%%
# Simulation

seqNet['spikemonOrd_Seq']
seqNet.run(duration)  # , report='text')
print ('ready...')

SequenceLearningExample.plot()

#%%
# You can now set the standaloneParams
# first print them in order to see, what we can change:
seqNet.printParams()

#%%
standaloneParams = OrderedDict([('duration', 0.33 * second),
                             ('sInpOrd1e_Seq_weight', 1.3),
                             ('sOrdMem1e_Seq_weight', 1.1),
                             ('sMemOrd1e_Seq_weight', 0.16),
                             ('sOrdOrd1e_Seq_weight', 1.04),
                             ('sMemMem1e_Seq_weight', 1.54),
                             ('sOrdOrd1i_Seq_weight', -1.95),
                             ('sMemOrd1i_Seq_weight', -0.384),
                             ('sCoSOrd1i_Seq_weight', -1.14),
                             ('sResOrd1i_Seq_weight', -1.44),
                             ('sResMem1i_Seq_weight', -2.6),
                             ('gOrd_Seq_refP', 1.7 * msecond),
                             ('gMem_Seq_refP', 2.3 * msecond),
                             ('gOrd_Seq_b', 80.5 * pamp),
                             ('gOrd_Seq_C', 281. * pfarad),
                             ('taugIi', 6. * msecond)])

standaloneParams = OrderedDict([('duration', 0.33 * second),
                 ('gOrd_Seq_refP', 1.7 * msecond),
                 ('gMem_Seq_refP', 2.3 * msecond),
                 ('sInpOrd1e_Seq_weight', 1.3),
                 ('sOrdMem1e_Seq_weight', 1.1),
                 ('sMemOrd1e_Seq_weight', 0.16),
                 ('sOrdOrd1e_Seq_weight', 1.04),
                 ('sMemMem1e_Seq_weight', 1.54),
                 ('sOrdOrd1i_Seq_weight', -1.95),
                 ('sMemOrd1i_Seq_weight', -0.384),
                 ('sCoSOrd1i_Seq_weight', -1.14),
                 ('sResOrd1i_Seq_weight', -1.44),
                 ('sResMem1i_Seq_weight', -2.6)])
#%%

seqNet.run(standaloneParams=standaloneParams)
#%%
SequenceLearningExample.plot()
