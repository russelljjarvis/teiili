#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:50:14 2017

This example shows mainly how the standalone code generation with changeable parameters works.
It simulates the core of the sequence learning architecture.

If you are using anaconda, you might need to update libgcc before the cpp code generation works correctly (conda install libgcc).
Version 5.2. works for me.


@author: alpha
"""
import numpy as np
import os
from collections import OrderedDict

from brian2 import ms, us, second, prefs, set_device, defaultclock, size, msecond

from teili.building_blocks.sequence_learning import SequenceLearning

from teili import NeuronEquationBuilder, SynapseEquationBuilder
from teili.core.network import TeiliNetwork
from teili.tools.cpptools import activate_standalone

neuron_model = NeuronEquationBuilder.import_eq('ExpAdaptIF', num_inputs=1)
synapse_model = SynapseEquationBuilder.import_eq('ReversalSynV')

standalone_dir = os.path.expanduser('~/SL_standalone')
prefs.codegen.target = 'numpy'

activate_standalone(directory=standalone_dir, build_on_run=False)

prefs.devices.cpp_standalone.openmp_threads = 4
prefs.devices.cpp_standalone.extra_make_args_unix = ["-j$(nproc)"]

defaultclock.dt = 100 * us

duration = 330 * ms
n_per_group = 8
num_elem = 3
num_ord_neurons = num_elem * n_per_group

sl_params = {'synInpOrd1e_weight': 1.3,
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

sequence_learning_example = SequenceLearning(name='Seq',
                                             neuron_eq_builder=neuron_model,
                                             synapse_eq_builder=synapse_model,
                                             block_params=sl_params,
                                             num_elements=num_elem,
                                             num_neurons_per_group=n_per_group,
                                             verbose=False)

# for key in SLGroups: print(key)

# Input to start sequence manually
ts_input = np.concatenate(
    (np.ones((n_per_group,), dtype=np.int) * 5, np.ones((n_per_group,), dtype=np.int) * 6,
     np.ones((n_per_group,), dtype=np.int) * 7
     )) * ms
ind_input = np.mod(np.arange(size(ts_input)), n_per_group)
sequence_learning_example.input_group.set_spikes(indices=ind_input, times=ts_input)

# CoS group
ts_cos = np.concatenate(
    (np.ones((n_per_group,), dtype=np.int) * 100, np.ones((n_per_group,), dtype=np.int) * 101,
     np.ones((n_per_group,), dtype=np.int) * 102, np.ones((n_per_group,), dtype=np.int) * 103,
     np.ones((n_per_group,), dtype=np.int) * 104, np.ones((n_per_group,), dtype=np.int) * 105,

     np.ones((n_per_group,), dtype=np.int) * 200, np.ones((n_per_group,), dtype=np.int) * 201,
     np.ones((n_per_group,), dtype=np.int) * 202, np.ones((n_per_group,), dtype=np.int) * 203,
     np.ones((n_per_group,), dtype=np.int) * 204, np.ones((n_per_group,), dtype=np.int) * 205
     )) * ms
ind_cos = np.mod(np.arange(size(ts_cos)), n_per_group)
sequence_learning_example.cos_group.set_spikes(indices=ind_cos, times=ts_cos)

# reset group
ts_reset = np.concatenate(
    (np.ones((n_per_group,), dtype=np.int) * 300, np.ones((n_per_group,), dtype=np.int) * 301,
     np.ones((n_per_group,), dtype=np.int) * 302, np.ones((n_per_group,), dtype=np.int) * 303,
     np.ones((n_per_group,), dtype=np.int) * 304, np.ones((n_per_group,), dtype=np.int) * 305,
     np.ones((n_per_group,), dtype=np.int) * 306, np.ones((n_per_group,), dtype=np.int) * 307
     )) * ms
ind_reset = np.mod(np.arange(size(ts_reset)), n_per_group)
sequence_learning_example.reset_group.set_spikes(indices=ind_reset, times=ts_reset)

seq_net = TeiliNetwork()
# seqNet = Network()
seq_net.add(sequence_learning_example, sequence_learning_example.monitors)

# This is how you add additional parameters that you want to change in the standalone run (you just have to find out their names...)
# taugIi in this case is valid for all neurons!
# Note that this is string replacement, so if you have another state variable that is called e.g. GammataugIi, this would also be replaced!
# seqNet.add_standalone_params(gOrd_Seq_b=0.0805*nA, taugIi=6*ms)

standalone_params = OrderedDict([('duration', 0.33 * second),
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
                                 # ('gOrd_Seq_refP', 1.7 * msecond),
                                 # ('gMem_Seq_refP', 2.3 * msecond),
                                 # ('gOrd_Seq_b', 80.5 * pamp),
                                 # ('gOrd_Seq_C', 281. * pfarad),
                                 # ('taugIi', 6. * msecond)
                                 ])
# standalone_params = {'duration': 0.33 * second}
seq_net.standalone_params = standalone_params
seq_net.build()

# %%
# Simulation

seq_net['spikemonOrd_Seq']
seq_net.run(duration * second, standaloneParams=standalone_params)  # , report='text')
print('ready...')

sequence_learning_example.plot(duration=duration)

# %%
# You can now set the standaloneParams
# First print them in order to see what we can change:
seq_net.print_params()

# %%


standalone_params = OrderedDict([('duration', 0.33 * second),
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
# %%

seq_net.run(standaloneParams=standalone_params)
# %%
sequence_learning_example.plot(duration=duration)

seq_net['gMem_Seq'].equation_builder.keywords
