"""
Created on Thu Jun 20 09:27:37 2019

@author: matteo

This module provides a hierarchical building block called OCTA.
            -Online Clustering of Temporal Activity-


Attributes:
    wta_params (dict): Dictionary of default parameters for wta.
    octa_params (dict): Dictionary of default parameters for wta.

"""

from teili.building_blocks.wta import WTA
from teili.core.groups import Neurons, Connections
from teili.stimuli.testbench import WTA_Testbench, OCTA_Testbench
from teili import TeiliNetwork as teiliNetwork
from teili.models.synapse_models import DPISyn, DPIstdp
from teili.models.neuron_models import DPI
from teili.tools.indexing import ind2xy, ind2events
from teili.tools.plotting import plot_spikemon_qt, plot_statemon_qt
from teili.tools.sorting import SortMatrix

# Load modified neuron and synapse models
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder ,\
print_neuron_model
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder ,\
print_synaptic_model


from teili.octa.tools.add_run_reg import *
from teili.octa.tools.add_mismatch import *
from teili.octa.tools.weight_init import weight_init

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph.Point import Point

from teili.models.parameters import octa_param 
from teili.models.parameters.octa_param import * 

import numpy as np
import time, os, sys
from tkinter import filedialog

from brian2 import ms, us, second, amp, pA, mA, nA
from brian2.units.allunits import radian
from brian2 import StateMonitor, SpikeMonitor, SpikeGeneratorGroup
from brian2 import prefs, defaultclock, check_units, implementation, set_device, device

from teili.models.synapse_models import DPIstdp


re_init_weights = True
inh_threshold = 'random'
adapt_threshold = False
expected_EI_syn = 350
mismatch = True
weight_regularization = True

save = True
debug = 2
save_directory = '/home/matteo/Documents/Repositories/OCTA/results/'

prefs.codegen.target = 'numpy'
np.random.seed(42)
# set_device('cpp_standalone', directory='~/Downloads/')
# device.reinit()
# device.activate('cpp_standalone', directory='~/Downloads/')

defaultclock.dt = 500 * us
Net = teiliNetwork()



testbench_c = WTA_Testbench()
testbench_p = WTA_Testbench()
testbench_stim = OCTA_Testbench()

#Neuron population sizes
num_neurons = octa_param.octaParams['num_neurons']
num_input_neurons = octa_param.octaParams[ 'num_input_neurons']


# other init parameters
duration = octa_param.octaParams['duration']
buffer_size = octa_param.octaParams['buffer_size']

##########################

# NeuronGroups
inputGroup = SpikeGeneratorGroup(N=num_input_neurons**2, indices=[], times=[]*ms)

# if (expected_EI_syn * 4) / num_neurons**4 <= 1.0:
#     wtaParams['EI_connection_probability'] = round((expected_EI_syn * 4) / num_neurons**4, 2)
# else:
#     wtaParams['EI_connection_probability'] = 1.0

num_inh_neurons_c = int(num_neurons**2/4) 
num_inh_neurons_p = int(num_input_neurons**2/4)


compressionWTA = WTA(name='compressionWTA', dimensions=2, 
                     neuron_eq_builder=octa_param.octa_neuron,
                     num_neurons=num_neurons, num_inh_neurons=num_inh_neurons_c,
                     num_input_neurons=num_input_neurons, num_inputs=2, 
                     block_params=octa_param.wtaParams, 
                     monitor=True)

predictionWTA = WTA(name='predictionWTA', dimensions=2,
                    neuron_eq_builder=octa_param.octa_neuron,
                    num_neurons=num_input_neurons, num_inh_neurons=num_inh_neurons_p,
                    num_input_neurons=num_neurons, num_inputs=2,
                    block_params=octa_param.wtaParams, 
                    monitor=True)
#%%

compressionWTA._groups['n_exc'] = Neurons(num_input_neurons**2, equation_builder=octa_param.octa_neuron(num_inputs=4),
                                                refractory=octa_param.wtaParams['rp_exc'],
                                                name=compressionWTA._groups['n_exc'].name)


inputSynapse = Connections(inputGroup, compressionWTA._groups['n_exc'],
                           equation_builder=octa_param.DPISyn,
                           method='euler',
                           name='inputSynapse')
inputSynapse.connect('i==j')
inputSynapse.weight = 3250.

compressionWTA._groups['s_inp_exc'] = Connections(compressionWTA._groups['spike_gen'], 
                                                   compressionWTA._groups['n_exc'],
                                                   equation_builder=octa_param.DPIstdp_gm,
                                                   method='euler', 
                                                   name=compressionWTA._groups['s_inp_exc'].name)

compressionWTA._groups['s_inp_exc'].connect('True')
compressionWTA._groups['s_inp_exc'].weight = octa_param.wtaParams['we_inp_exc']
compressionWTA._groups['s_inp_exc'].taupre = octa_param.octaParams['tau_stdp']
compressionWTA._groups['s_inp_exc'].taupost = octa_param.octaParams['tau_stdp']


compressionWTA._groups['s_inp_exc'].w_plast = weight_init(compressionWTA._groups['s_inp_exc'], 
                                                           dist_param=octa_param.octaParams['dist_param_init'], 
                                                           scale=octa_param.octaParams['scale_init'],
                                                           distribution=octa_param.octaParams['distribution'])
#%%
compressionWTA._groups['s_exc_exc'] = Connections(compressionWTA._groups['n_exc'], 
                                                   compressionWTA._groups['n_exc'],
                                                   equation_builder=DPIstdp(),
                                                   method='euler', 
                                                   name=compressionWTA._groups['s_exc_exc'].name)

'''

compressionWTA.Groups['synWTAWTA1e'] = Connections(compressionWTA.Groups['gWTAGroup'], 
                                                   compressionWTA.Groups['gWTAGroup'],
                                                   equation_builder=DPIstdp(),
                                                   method='euler',
                                                   name=compressionWTA.Groups['synWTAWTA1e'].name)
'''
#%%

compressionWTA._groups['s_exc_exc'].connect('True')
compressionWTA._groups['s_exc_exc'].weight = octa_param.wtaParams['we_exc_exc']

compressionWTA._groups['s_exc_exc'].w_plast = weight_init(compressionWTA._groups['s_exc_exc'], 
                                                           dist_param=octa_param.octaParams['dist_param_init'], 
                                                           scale=octa_param.octaParams['scale_init'],
                                                           distribution=octa_param.octaParams['distribution'])
# Inhibitory plasticity
#%%
compressionWTA._groups['s_inh_exc'] = Connections(compressionWTA._groups['n_inh'],
                                                   compressionWTA._groups['n_exc'],
                                                   equation_builder=DPIadp,
                                                   method='euler',
                                                   name=compressionWTA._groups['s_inh_exc'].name)


compressionWTA._groups['s_inh_exc'].connect('True')
compressionWTA._groups['s_inh_exc'].weight = octa_param.wtaParams['wi_inh_exc']

compressionWTA._groups['s_inh_exc'].variance_th = np.random.uniform(low=octa_param.octaParams['variance_th_c']-0.1,
                                                                     high=octa_param.octaParams['variance_th_c']+0.1,
                                                                     size=len(compressionWTA._groups['s_inh_exc']))

if inh_threshold=='fixed':
    compressionWTA._groups['s_inh_exc'].variance_th = np.clip(compressionWTA._groups['s_inh_exc'].variance_th, 
                                                               octa_param.octaParams['variance_th_c'], 
                                                               octa_param.octaParams['variance_th_c'])

compressionWTA._groups['s_inh_exc'].w_plast = weight_init(compressionWTA._groups['s_inh_exc'], 
                                                           dist_param=octa_param.octaParams['dist_param_init'], 
                                                           scale=octa_param.octaParams['scale_init'],
                                                           distribution=octa_param.octaParams['distribution'])


predictionWTA._groups['s_inh_exc'] = Connections(predictionWTA._groups['n_inh'],
                                                  predictionWTA._groups['n_exc'],
                                                  equation_builder=octa_param.DPIadp,
                                                  method='euler',
                                                  name=predictionWTA._groups['s_inh_exc'].name)

predictionWTA._groups['s_inh_exc'].connect('True')
predictionWTA._groups['s_inh_exc'].weight = octa_param.wtaParams['wi_inh_exc']

predictionWTA._groups['s_inh_exc'].variance_th =np.random.uniform(low=octa_param.octaParams['variance_th_p']-0.1,
                                                                   high=octa_param.octaParams['variance_th_p']+0.1,
                                                                   size=len(predictionWTA._groups['s_inh_exc']))
if inh_threshold=='fixed':
    predictionWTA._groups['s_inh_exc'].variance_th = np.clip(predictionWTA._groups['s_inh_exc'].variance_th, 
                                                              octa_param.octaParams['variance_th_p'], 
                                                              octa_param.octaParams['variance_th_p'])

predictionWTA._groups['s_inh_exc'].w_plast = weight_init(predictionWTA._groups['s_inh_exc'], 
                                                          dist_param=octa_param.octaParams['dist_param_init'], 
                                                          scale=octa_param.octaParams['scale_init'],
                                                          distribution=octa_param.octaParams['distribution'])

predictionWTA._groups['s_inp_exc'] = Connections(compressionWTA._groups['n_exc'], 
                                                  predictionWTA._groups['n_exc'],
                                                  equation_builder=DPIstdp,
                                                  method='euler', 
                                                  name=predictionWTA._groups['s_exc_exc'].name)

predictionWTA._groups['s_inp_exc'].connect('True')
predictionWTA._groups['s_inp_exc'].weight = octa_param.wtaParams['we_inp_exc']
predictionWTA._groups['s_inp_exc'].taupre = octa_param.octaParams['tau_stdp']
predictionWTA._groups['s_inp_exc'].taupost = octa_param.octaParams['tau_stdp']

predictionWTA._groups['s_inp_exc'].w_plast = weight_init(predictionWTA._groups['s_inp_exc'], 
                                                          dist_param=octa_param.octaParams['dist_param_init'], 
                                                          scale=octa_param.octaParams['scale_init'],
                                                          distribution=octa_param.octaParams['distribution'])


predictionWTA._groups['s_exc_exc'] = Connections(predictionWTA._groups['n_exc'], 
                                                  predictionWTA._groups['n_exc'],
                                                  equation_builder=DPIstdp,
                                                  method='euler',
                                                  name=predictionWTA._groups['s_exc_exc'].name)

predictionWTA._groups['s_exc_exc'].connect('True')
predictionWTA._groups['s_exc_exc'].weight = octa_param.wtaParams['we_exc_exc']

predictionWTA._groups['s_exc_exc'].w_plast = weight_init(predictionWTA._groups['s_exc_exc'], 
                                                          dist_param=octa_param.octaParams['dist_param_init'], 
                                                          scale=octa_param.octaParams['scale_init'],
                                                          distribution=octa_param.octaParams['distribution'])

error_connection = Connections(compressionWTA._groups['spike_gen'], 
                               predictionWTA._groups['n_exc'],
                               equation_builder=octa_param.DPIstdp_gm,
                               method='euler', 
                               name='error_connection')

error_connection.connect('True')
error_connection.weight = octa_param.wtaParams['we_inp_exc']
error_connection.taupre = octa_param.octaParams['tau_stdp']
error_connection.taupost = octa_param.octaParams['tau_stdp']

error_connection.w_plast = weight_init(error_connection, 
                                       dist_param=octa_param.octaParams['dist_param_init'], 
                                       scale=octa_param.octaParams['scale_init'],
                                       distribution=octa_param.octaParams['distribution'])

# PREDICTION SYNAPSES AND SPIKE TIME DEPENDENT GAINMODULATION
# We add a state variable to the Spike generator 

# compressionWTA.inputGroup.Ipred = 1.0
# Then we create a synapse from L6 (predictionWTA) to L4 (inputSpike generator)
# using the updated DPI eq DPISyn_gm gm = gain modulation

prediction_connection = Connections(predictionWTA._groups['n_exc'],
                                    compressionWTA._groups['spike_gen'],
                                    equation_builder=octa_param.SynSTDGM,
                                    method='euler',
                                    name='prediction_connection')
prediction_connection.connect(True)

prediction_connection.Ipred_plast = np.zeros((len(prediction_connection)))

# Set learning rate
compressionWTA._groups['s_inp_exc'].dApre = octa_param.octaParams['learning_rate']
compressionWTA._groups['s_exc_exc'].dApre = octa_param.octaParams['learning_rate']
predictionWTA._groups['s_inp_exc'].dApre = octa_param.octaParams['learning_rate']
predictionWTA._groups['s_exc_exc'].dApre = octa_param.octaParams['learning_rate']

error_connection.dApre = octa_param.octaParams['learning_rate']
prediction_connection.dApre = octa_param.octaParams['learning_rate']

compressionWTA._groups['s_inh_exc'].inh_learning_rate = octa_param.octaParams['inh_learning_rate']
predictionWTA._groups['s_inh_exc'].inh_learning_rate = octa_param.octaParams['inh_learning_rate']

#%%

# Adding mismatch
if mismatch:
    add_device_mismatch(compressionWTA._groups['gWTAInpGroup'], group_type='neuron', seed=42)
    add_device_mismatch(compressionWTA._groups['gWTAGroup'], group_type='neuron', seed=43)
    add_device_mismatch(compressionWTA._groups['gWTAInhGroup'], group_type='neuron', seed=44)
    
    add_device_mismatch(compressionWTA._groups['synInpWTA1e'], group_type='synapse', seed=45)
    add_device_mismatch(compressionWTA._groups['synWTAWTA1e'], group_type='synapse', seed=46)
    add_device_mismatch(compressionWTA._groups['synWTAInh1e'], group_type='synapse', seed=47)
    add_device_mismatch(compressionWTA._groups['synInhWTA1i'], group_type='synapse', seed=48)
    
    add_device_mismatch(predictionWTA._groups['gWTAGroup'], group_type='neuron', seed=49)
    add_device_mismatch(predictionWTA._groups['gWTAInhGroup'], group_type='neuron', seed=50)
    
    add_device_mismatch(predictionWTA._groups['synInpWTA1e'], group_type='synapse', seed=51)
    add_device_mismatch(predictionWTA._groups['synWTAWTA1e'], group_type='synapse', seed=52)
    add_device_mismatch(predictionWTA._groups['synWTAInh1e'], group_type='synapse', seed=53)
    add_device_mismatch(predictionWTA._groups['synInhWTA1i'], group_type='synapse', seed=1337)
    
    add_device_mismatch(error_connection, group_type='synapse', seed=54)
    


# RUN REGULARLY FUNCTIONS ###############
# WEIGHT DECAY
if octaParams['weight_decay'] == 'global':
    add_weight_decay(compressionWTA.Groups['synInpWTA1e'],
                     decay=octaParams['weight_decay'],
                     learning_rate=octaParams['learning_rate'])

    add_weight_decay(compressionWTA.Groups['synWTAWTA1e'],
                     decay=octaParams['weight_decay'],
                     learning_rate=octaParams['learning_rate'])

    add_weight_decay(predictionWTA.Groups['synInpWTA1e'],
                     decay=octaParams['weight_decay'],
                     learning_rate=octaParams['learning_rate'])

    add_weight_decay(predictionWTA.Groups['synWTAWTA1e'],
                     decay=octaParams['weight_decay'],
                     learning_rate=octaParams['learning_rate'])
    
#     add_weight_decay(compressionWTA.Groups['synInhWTA1i'],
#                      decay=octaParams['weight_decay'],
#                      learning_rate=octaParams['inh_learning_rate'])

#     add_weight_decay(predictionWTA.Groups['synInhWTA1i'],
#                      decay=octaParams['weight_decay'],
#                      learning_rate=octaParams['inh_learning_rate'])

    add_weight_decay(error_connection,
                     decay=octaParams['weight_decay'],
                     learning_rate=octaParams['learning_rate'])

    add_pred_weight_decay(prediction_connection,
                          decay=octaParams['weight_decay'],
                          learning_rate=octaParams['learning_rate'])
if adapt_threshold:
    add_adaptive_threshold(compressionWTA.Groups['gWTAGroup'],
                           update_step=0.5*pA,
                           decay_time=200*ms)
    add_adaptive_threshold(compressionWTA.Groups['gWTAInhGroup'],
                           update_step=0.0*pA,
                           decay_time=200*ms)
    add_adaptive_threshold(predictionWTA.Groups['gWTAGroup'],
                           update_step=0.5*pA,
                           decay_time=200*ms)
    add_adaptive_threshold(predictionWTA.Groups['gWTAInhGroup'],
                           update_step=0.0*pA,
                           decay_time=200*ms)

    compressionWTA.Groups['gWTAGroup'].adaptive_threshold = 1.0 * pA
    compressionWTA.Groups['gWTAInhGroup'].adaptive_threshold = 0.0 * pA
    predictionWTA.Groups['gWTAGroup'].adaptive_threshold = 1.0 * pA
    predictionWTA.Groups['gWTAInhGroup'].adaptive_threshold = 0.0 * pA
else:
    compressionWTA.Groups['gWTAGroup'].adaptive_threshold = 0.0 * pA
    compressionWTA.Groups['gWTAInhGroup'].adaptive_threshold = 0.0 * pA
    predictionWTA.Groups['gWTAGroup'].adaptive_threshold = 0.0 * pA
    predictionWTA.Groups['gWTAInhGroup'].adaptive_threshold = 0.0 * pA

# RE-INIT WEIGHT MATRICES
if re_init_weights:
    add_re_init_weights(compressionWTA.Groups['synInpWTA1e'], 
                        re_init_threshold=octaParams['re_init_threshold'],
                        dist_param_re_init=octaParams['dist_param_re_init'], 
                        scale_re_init=octaParams['scale_re_init'],
                        distribution=octaParams['distribution'])

#     add_re_init_weights(compressionWTA.Groups['synInhWTA1i'], 
#                         re_init_threshold=octaParams['re_init_threshold'],
#                         dist_param_re_init=octaParams['dist_param_re_init'], 
#                         scale_re_init=octaParams['scale_re_init'],
#                         distribution=octaParams['distribution'])

    add_re_init_weights(compressionWTA.Groups['synWTAWTA1e'], 
                        re_init_threshold=octaParams['re_init_threshold'],
                        dist_param_re_init=octaParams['dist_param_re_init'], 
                        scale_re_init=octaParams['scale_re_init'],
                        distribution=octaParams['distribution'])

    add_re_init_weights(predictionWTA.Groups['synInpWTA1e'], 
                        re_init_threshold=octaParams['re_init_threshold'],
                        dist_param_re_init=octaParams['dist_param_re_init'], 
                        scale_re_init=octaParams['scale_re_init'],
                        distribution=octaParams['distribution'])

#     add_re_init_weights(predictionWTA.Groups['synInhWTA1i'], 
#                         re_init_threshold=octaParams['re_init_threshold'],
#                         dist_param_re_init=octaParams['dist_param_re_init'], 
#                         scale_re_init=octaParams['scale_re_init'],
#                         distribution=octaParams['distribution'])

    add_re_init_weights(predictionWTA.Groups['synWTAWTA1e'], 
                        re_init_threshold=octaParams['re_init_threshold'],
                        dist_param_re_init=octaParams['dist_param_re_init'], 
                        scale_re_init=octaParams['scale_re_init'],
                        distribution=octaParams['distribution'])

    add_re_init_weights(error_connection, 
                        re_init_threshold=octaParams['re_init_threshold'],
                        dist_param_re_init=octaParams['dist_param_re_init'], 
                        scale_re_init=octaParams['scale_re_init'],
                        distribution=octaParams['distribution'])

    add_re_init_ipred(prediction_connection, 
                      re_init_threshold=octaParams['re_init_threshold'])
#                       dist_param_re_init=octaParams['dist_param_re_init'], 
#                       scale_re_init=octaParams['scale_re_init'],
#                       distribution=octaParams['distribution'])


# INHIBITORY PLASTICITY
add_activity_proxy(compressionWTA.Groups['gWTAGroup'],
                   buffer_size=octaParams['buffer_size_plast'],
                   decay=octaParams['decay'])

add_activity_proxy(predictionWTA.Groups['gWTAGroup'],
                   buffer_size=octaParams['buffer_size_plast'],
                   decay=octaParams['decay'])

# WEIGHT REGULARIZATION
if weight_regularization:
    add_weight_regularization(compressionWTA.Groups['synInpWTA1e'],
                              buffer_size=octaParams['buffer_size'])
    add_weight_regularization(error_connection,
                              buffer_size=octaParams['buffer_size'])



# LOADING INPUT AND ADDING NOISE 
testbench_stim.rotating_bar(length=10, nrows=10, 
                            direction='cw', 
                            ts_offset=3, angle_step=10, 
                            noise_probability=0.2, repetitions=octaParams['revolutions'], debug=False)

inputGroup.set_spikes(indices=testbench_stim.indices, times=testbench_stim.times * ms)


testbench_c.background_noise(num_neurons=num_neurons, rate=10)
testbench_p.background_noise(num_neurons=num_input_neurons, rate=10)

noise_syn_c = Connections(testbench_c.noise_input, 
                        compressionWTA,
                        equation_builder=DPISyn(), 
                        name="noise_syn",)

noise_syn_c.connect("i==j")
noise_syn_c.weight = octaParams['noise_weight']


noise_syn_p = Connections(testbench_p.noise_input, 
                          predictionWTA,
                          equation_builder=DPISyn(), 
                          name="noise_syn_p",)

noise_syn_p.connect("i==j")
noise_syn_p.weight = octaParams['noise_weight']


