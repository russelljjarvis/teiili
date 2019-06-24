#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:27:37 2019

@author: matteo
"""

from teili.building_blocks.wta import WTA
from teili.models.neuron_models import DPI
from brian2 import us, ms, pA, nA, prefs,\
        SpikeMonitor, StateMonitor,\
        SpikeGeneratorGroup

from teili import TeiliNetwork

from teili.models.synapse_models import DPIstdp, DPISyn

Net= TeiliNetwork()
wta_params = {'we_inp_exc': 1.5,
              'we_exc_inh': 1,
              'wi_inh_exc': -1,
              'we_exc_exc': 0.5,
              'sigm': 3,
              'rp_exc': 3 * ms,
              'rp_inh': 1 * ms,
              'ei_connection_probability': 1,
              'ie_connection_probability': 1,
              'ii_connection_probability': 0}

prefs.codegen.target = 'numpy'

num_neurons = 7

num_input_neurons = 10
my_wta = WTA(name='my_wta', dimensions=1,
             neuron_eq_builder=DPI,
             num_neurons=num_neurons, num_inh_neurons=int(num_neurons**2/4),
             num_input_neurons=num_input_neurons, num_inputs=4,
             block_params=wta_params,
             monitor=True)

Net.add(
        my_wta._groups['spike_gen'], 
        my_wta._groups['n_exc'], 
        my_wta._groups['n_inh'], 
        my_wta._groups['s_inp_exc'], 
        my_wta._groups['s_exc_exc'], 
        my_wta._groups['s_exc_inh'],              
        my_wta._groups['s_inh_exc'],    
         my_wta.monitors['spikemon_exc'], 
         my_wta.monitors['spikemon_inh'],
         my_wta.monitors['spikemon_inp'])

Net.run(10*ms)         
         
