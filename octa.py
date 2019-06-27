#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:58:50 2019

@author: matteo


This module provides a hierarchical building block called OCTA.
            -Online Clustering of Temporal Activity-

Attributes:
    wta_params (dict): Dictionary of default parameters for wta.
    octa_params (dict): Dictionary of default parameters for wta.

"""
from brian2 import ms, us
from brian2 import StateMonitor, SpikeMonitor, SpikeGeneratorGroup
from brian2 import prefs, defaultclock

import matplotlib.pyplot as plt
import numpy as np
import time, os, sys

from teili.building_blocks.building_block import BuildingBlock
from teili.building_blocks.wta import WTA
from teili.core.groups import Neurons, Connections
from teili.stimuli.testbench import WTA_Testbench, OCTA_Testbench
from teili import TeiliNetwork as teiliNetwork
from teili.models.synapse_models import DPISyn, DPIstdp
from teili.tools.sorting import SortMatrix
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder ,\
print_neuron_model
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder ,\
print_synaptic_model

from teili.tools.octa_tools import octa_param
from teili.tools.octa_tools.octa_param import *

from teili.tools.octa_tools.weight_init import weight_init 

import teili.tools.tags_parameters as tags_parameters
from teili.tools.octa_tools.octa_tools import add_decay_weight, add_weight_re_init, add_weight_re_init_ipred,\
add_proxy_activity, add_regulatization_weight, add_weight_pred_decay, add_bb_mismatch
from teili.tools.octa_tools.lock_and_load import  save_monitor, load_monitor,\
save_weights, load_weights
prefs.codegen.target = 'numpy'

mismatch = True
weight_regularization = True

save_directory = '/home/matteo/Documents/Repositories/OCTA/results/'

prefs.codegen.target = 'numpy'


defaultclock.dt = 500 * us

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






inputGroup = SpikeGeneratorGroup(N=num_input_neurons**2, indices=[], times=[]*ms)
num_inh_neurons_c = int(num_neurons**2/4) 
num_inh_neurons_p = int(num_input_neurons**2/4)

compressionWTA = WTA(name='compressionWTA', dimensions=2, 
                     neuron_eq_builder=octa_param.octa_neuron,
                     num_neurons=num_neurons, num_inh_neurons=num_inh_neurons_c,
                     num_input_neurons=num_input_neurons, num_inputs=4, 
                     block_params=octa_param.wtaParams,
                     monitor=False)

predictionWTA = WTA(name='predictionWTA', dimensions=2,
                    neuron_eq_builder=octa_param.octa_neuron,
                    num_neurons=num_input_neurons, num_inh_neurons=num_inh_neurons_p,
                    num_input_neurons=num_neurons, num_inputs=4,
                    block_params=octa_param.wtaParams, 
                    monitor=False)



compressionWTA._groups['spike_gen'] = Neurons(num_input_neurons**2, equation_builder=octa_param.octa_neuron(num_inputs=3),
                                                refractory=octa_param.wtaParams['rp_exc'],
                                                name=compressionWTA._groups['spike_gen'].name)




compressionWTA._set_tags(tags_parameters.basic_tags_n_sg, compressionWTA._groups['spike_gen'])
 

inputSynapse = Connections(inputGroup, compressionWTA._groups['spike_gen'],
                           equation_builder=DPISyn(),
                           method='euler',
                           name='inputSynapse')


inputSynapse.connect('i==j')
inputSynapse.weight = 3250.


compressionWTA._groups['s_inp_exc'] = Connections(compressionWTA._groups['spike_gen'], 
                                                   compressionWTA._groups['n_exc'],
                                                   equation_builder=octa_param.DPIstdp_gm,
                                                   method='euler', 
                                                   name=compressionWTA._groups['s_inp_exc'].name)

compressionWTA._set_tags(tags_parameters.basic_tags_s_inp_exc, compressionWTA._groups['s_inp_exc'])


compressionWTA._groups['s_inp_exc'].connect('True')
compressionWTA._groups['s_inp_exc'].weight = octa_param.wtaParams['we_inp_exc']
compressionWTA._groups['s_inp_exc'].taupre = octa_param.octaParams['tau_stdp']
compressionWTA._groups['s_inp_exc'].taupost = octa_param.octaParams['tau_stdp']


compressionWTA._groups['s_inp_exc'].w_plast = weight_init(compressionWTA._groups['s_inp_exc'], 
                                                           dist_param=octa_param.octaParams['dist_param_init'], 
                                                           scale=octa_param.octaParams['scale_init'],
                                                           distribution=octa_param.octaParams['distribution'])


compressionWTA._groups['s_exc_exc'] = Connections(compressionWTA._groups['n_exc'], 
                                                   compressionWTA._groups['n_exc'],
                                                   equation_builder=DPIstdp(),
                                                   method='euler', 
                                                   name=compressionWTA._groups['s_exc_exc'].name)

compressionWTA._set_tags(tags_parameters.basic_tags_s_exc_exc, compressionWTA._groups['s_exc_exc'])

compressionWTA._groups['s_exc_exc'].connect('True')
compressionWTA._groups['s_exc_exc'].weight = octa_param.wtaParams['we_exc_exc']

compressionWTA._groups['s_exc_exc'].w_plast = weight_init(compressionWTA._groups['s_exc_exc'], 
                                                           dist_param=octa_param.octaParams['dist_param_init'], 
                                                           scale=octa_param.octaParams['scale_init'],
                                                           distribution=octa_param.octaParams['distribution'])
# Inhibitory plasticity

compressionWTA._groups['s_inh_exc'] = Connections(compressionWTA._groups['n_inh'],
                                                   compressionWTA._groups['n_exc'],
                                                   equation_builder=DPIadp,
                                                   method='euler',
                                                   name=compressionWTA._groups['s_inh_exc'].name)

compressionWTA._set_tags(tags_parameters.basic_tags_s_inh_exc, compressionWTA._groups['s_inh_exc'])


w_init_list= [compressionWTA._groups['s_exc_exc']] +\
            [compressionWTA._groups['s_inp_exc']] +\ 
            [compressionWTA._groups['s_inh_exc']] + \
            [predictionWTA._groups['s_inh_exc']] +\
            [predictionWTA._groups['s_inp_exc']] +\
            [error_connection]
            




compressionWTA._groups['s_inh_exc'].connect('True')
compressionWTA._groups['s_inh_exc'].weight = octa_param.wtaParams['wi_inh_exc']


compressionWTA._groups['s_inh_exc'].variance_th = np.random.uniform(low=octa_param.octaParams['variance_th_c']-0.1,
                                                                     high=octa_param.octaParams['variance_th_c']+0.1,
                                                                     size=len(compressionWTA._groups['s_inh_exc']))

compressionWTA._groups['s_inh_exc'].w_plast = weight_init(compressionWTA._groups['s_inh_exc'], 
                                                           dist_param=octa_param.octaParams['dist_param_init'], 
                                                           scale=octa_param.octaParams['scale_init'],
                                                           distribution=octa_param.octaParams['distribution'])


predictionWTA._groups['s_inh_exc'] = Connections(predictionWTA._groups['n_inh'],
                                                  predictionWTA._groups['n_exc'],
                                                  equation_builder=octa_param.DPIadp,
                                                  method='euler',
                                                  name=predictionWTA._groups['s_inh_exc'].name)

compressionWTA._set_tags(tags_parameters.basic_tags_s_inh_exc, predictionWTA._groups['s_inh_exc'])


predictionWTA._groups['s_inh_exc'].connect('True')
predictionWTA._groups['s_inh_exc'].weight = octa_param.wtaParams['wi_inh_exc']

predictionWTA._groups['s_inh_exc'].variance_th =np.random.uniform(low=octa_param.octaParams['variance_th_p']-0.1,
                                                                   high=octa_param.octaParams['variance_th_p']+0.1,
                                                                   size=len(predictionWTA._groups['s_inh_exc']))

predictionWTA._groups['s_inh_exc'].w_plast = weight_init(predictionWTA._groups['s_inh_exc'], 
                                                          dist_param=octa_param.octaParams['dist_param_init'], 
                                                          scale=octa_param.octaParams['scale_init'],
                                                          distribution=octa_param.octaParams['distribution'])

predictionWTA._groups['s_inp_exc'] = Connections(compressionWTA._groups['n_exc'], 
                                                  predictionWTA._groups['n_exc'],
                                                  equation_builder=DPIstdp,
                                                  method='euler', 
                                                  name=predictionWTA._groups['s_inp_exc'].name)
predictionWTA._set_tags(tags_parameters.basic_tags_s_inp_exc, predictionWTA._groups['s_inp_exc'])
predictionWTA._groups['s_inp_exc']._tags['sign'] = 'exc'
predictionWTA._groups['s_inp_exc']._tags['bb_type'] = 'octa'
predictionWTA._groups['s_inp_exc']._tags['connection_type'] = 'ff'



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

compressionWTA._set_tags(tags_parameters.basic_tags_s_exc_exc, predictionWTA._groups['s_exc_exc'])

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

#error_connection._set_tags(tags_parameters.basic_tags_empty, error_connection)

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


add_bb_mismatch(compressionWTA)
add_bb_mismatch(predictionWTA)
error_connection.add_mismatch(octa_param.mismatch_synap_param, seed= 42)

weight_decay_group =  [compressionWTA._groups['s_inp_exc']] + \
     [compressionWTA._groups['s_exc_exc']] +\
    [predictionWTA._groups['s_inp_exc']] + \
    [predictionWTA._groups['s_exc_exc']] + \
    [error_connection]
add_decay_weight( weight_decay_group, octa_param.octaParams['weight_decay'], 
                 octa_param.octaParams['learning_rate'] )    


pred_weight_decay_group = [prediction_connection]
add_weight_pred_decay( pred_weight_decay_group, octa_param.octaParams['weight_decay'],  octa_param.octaParams['learning_rate'] )    


weight_re_init_group= [compressionWTA._groups['s_inp_exc']] + \
    [compressionWTA._groups['s_exc_exc']] + \
    [predictionWTA._groups['s_inp_exc']] + \
    [predictionWTA._groups['s_exc_exc']] + \
    [error_connection]

add_weight_re_init(weight_re_init_group,re_init_threshold=octa_param.octaParams['re_init_threshold'],
                        dist_param_re_init=octa_param.octaParams['dist_param_re_init'], 
                        scale_re_init=octa_param.octaParams['scale_re_init'],
                        distribution=octa_param.octaParams['distribution'])


weight_re_init_ipred_group = [prediction_connection]
add_weight_re_init_ipred(weight_re_init_ipred_group, re_init_threshold=octa_param.octaParams['re_init_threshold'] )


weight_regularization_group = [compressionWTA._groups['s_inp_exc']] + \
                                [error_connection]
add_regulatization_weight(weight_regularization_group, buffer_size=octa_param.octaParams['buffer_size'] )


activity_proxy_group= [compressionWTA._groups['n_exc']] + [predictionWTA._groups['n_exc']]
add_proxy_activity(activity_proxy_group,  buffer_size=octa_param.octaParams['buffer_size_plast'],
                   decay=octa_param.octaParams['decay'])


#%%
# LOADING INPUT AND ADDING NOISE 
testbench_stim.rotating_bar(length=10, nrows=10, 
                            direction='cw', 
                            ts_offset=3, angle_step=10, 
                            noise_probability=0.2, repetitions=octa_param.octaParams['revolutions'], debug=False)

inputGroup.set_spikes(indices=testbench_stim.indices, times=testbench_stim.times * ms)


testbench_c.background_noise(num_neurons=num_neurons, rate=10)
testbench_p.background_noise(num_neurons=num_input_neurons, rate=10)

noise_syn_c_exc = Connections(testbench_c.noise_input, 
                        compressionWTA._groups['n_exc'],
                        equation_builder=DPISyn(), 
                        name="noise_syn_c_exc")
#
noise_syn_c_exc.connect("i==j")
noise_syn_c_exc.weight = octa_param.octaParams['noise_weight']

noise_syn_p_exc = Connections(testbench_c.noise_input, 
                        predictionWTA._groups['n_exc'],
                        equation_builder=DPISyn(), 
                        name="noise_syn_p_exc")



noise_syn_p_exc.connect("i==j")
noise_syn_p_exc.weight = octa_param.octaParams['noise_weight']

compressionWTA.monitors['statemon_exc']= StateMonitor(compressionWTA._groups['n_exc'], ('Imem', 'Iin'), record=True,
                                   name= 'statemon_exc')
compressionWTA.monitors['spikemon_inh']=  SpikeMonitor(compressionWTA._groups['n_inh'], 
                      name='spikemon_inh')
compressionWTA.monitors['spikemon_exc'] = SpikeMonitor(compressionWTA._groups['n_exc'], 
                      name='spikemon_exc')
compressionWTA.monitors['spikemon_inp'] = SpikeMonitor(compressionWTA._groups['spike_gen'], 
                      name='spikemon_inp')


predictionWTA.monitors['statemon_exc']= StateMonitor(predictionWTA._groups['n_exc'], ('Imem', 'Iin'), record=True,
                                   name='statemon_exc_p')
predictionWTA.monitors['spikemon_inh']=  SpikeMonitor(predictionWTA._groups['n_inh'], 
                      name='spikemon_inh_p')
predictionWTA.monitors['spikemon_exc'] = SpikeMonitor(predictionWTA._groups['n_exc'], 
                      name='spikemon_exc_p')



_groups = {
        'comp_n_exc' : compressionWTA._groups['n_exc'],
        'comp_n_inh' : compressionWTA._groups['n_inh'] ,
        'comp_n_spike_gen' :  compressionWTA._groups['spike_gen'],
        'comp_s_inp_exc':  compressionWTA._groups['s_inp_exc'],
        'comp_s_exc_exc': compressionWTA._groups['s_exc_exc'],
        'comp_s_exc_inh': compressionWTA._groups['s_exc_inh'],
        'comp_s_inh_exc': compressionWTA._groups['s_inh_exc'],
        'pred_n_exc' : predictionWTA._groups['n_exc'],
        'pred_n_inh' :  predictionWTA._groups['n_inh'],
        'pred_s_inp_exc': predictionWTA._groups['s_inp_exc'],
        'pred_s_exc_exc': predictionWTA._groups['s_exc_exc'],
        'pred_s_exc_inh': predictionWTA._groups['s_exc_inh'],
        'pred_s_inh_exc':  predictionWTA._groups['s_inh_exc'],
        'error_connection': error_connection,
        'prediction_connection' : prediction_connection,
        'inputGroup' : inputGroup,
        'inputSynapse': inputSynapse,
        'pred_noise_syn_exc' : noise_syn_p_exc,
        'comp_noise_syn_exc': noise_syn_c_exc
        
        
        }



monitors = {
         'comp_spikemon_exc' : compressionWTA.monitors['spikemon_exc'], 
         'comp_spikemon_inh' :compressionWTA.monitors['spikemon_inh'],
         'comp_spikemon_inp' :compressionWTA.monitors['spikemon_inp'],                 
         'comp_statemon_exc' :compressionWTA.monitors['statemon_exc'],
         
         'pred_spikemon_exc' : predictionWTA.monitors['spikemon_exc'], 
         'pred_spikemon_inh' :predictionWTA.monitors['spikemon_inh'],
         'pred_statemon_exc' :predictionWTA.monitors['statemon_exc'], 
         
         }

Net = teiliNetwork()


Net.add(
        inputGroup, inputSynapse,
         error_connection, prediction_connection,
         noise_syn_p_exc, noise_syn_c_exc,
         testbench_c.noise_input, testbench_p.noise_input,
        predictionWTA._groups['spike_gen'], 
        predictionWTA._groups['n_exc'], 
        predictionWTA._groups['n_inh'], 
        predictionWTA._groups['s_inp_exc'], 
        predictionWTA._groups['s_exc_exc'], 
        predictionWTA._groups['s_exc_inh'],              
        predictionWTA._groups['s_inh_exc'],              
         predictionWTA.monitors['statemon_exc'],
         predictionWTA.monitors['spikemon_inh'],
         predictionWTA.monitors['spikemon_exc'],
        compressionWTA._groups['spike_gen'], 
        compressionWTA._groups['n_exc'], 
        compressionWTA._groups['n_inh'], 
        compressionWTA._groups['s_inp_exc'], 
        compressionWTA._groups['s_exc_exc'], 
       compressionWTA._groups['s_exc_inh'],              
        compressionWTA._groups['s_inh_exc'],    
         compressionWTA.monitors['spikemon_exc'], 
         compressionWTA.monitors['spikemon_inh'],
         compressionWTA.monitors['spikemon_inp'],                 
         compressionWTA.monitors['statemon_exc'],          


         )


octa_param.octaParams['duration'] = np.max(testbench_stim.times)
Net.run(octa_param.octaParams['duration'] * ms, report='text')

#%%
compressionWTA._groups['s_exc_exc'].w_plast.shape
octa_tools.save_weights(compressionWTA._groups['s_exc_exc'].w_plast, filename='rec_c_weights_last', 
                path='/home/matteo/Documents/Repositories/teili_devBB/teili/teili/octa')
octa_tools.save_monitor(compressionWTA.monitors['spikemon_exc'], filename='spikemon_compressionWTA', path = '/home/matteo/Documents/Repositories/teili_devBB/teili/teili/octa/')
s = SortMatrix(nrows=49, ncols=49, filename='/home/matteo/Documents/Repositories/teili_devBB/teili/teili/octa26_06_2019/26_06_12_54_rec_c_weights_last.npy', axis=1)
mon = load_monitor(filename='teili/octa/26_06_2019/26_06_12_54_spikemon_compressionWTA.npy')

mon.i = np.asarray([np.where(np.asarray(s.permutation) == int(i))[0][0] for i in mon.i])
plt.plot(mon.i, '.k')

