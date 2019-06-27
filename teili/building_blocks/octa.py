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

from teili.tools.octa_tools.octa_param import wtaParameters, octaParameters, octa_neuron,\
DPIadp, DPIstdp_gm 
from teili.tools.octa_tools.weight_init import weight_init 

import teili.tools.tags_parameters as tags_parameters
from teili.tools.octa_tools.octa_tools import add_decay_weight, add_weight_re_init, add_weight_re_init_ipred,\
add_proxy_activity, add_regulatization_weight, add_weight_pred_decay, add_bb_mismatch
from teili.tools.octa_tools.lock_and_load import  save_monitor, load_monitor,\
save_weights, load_weights

prefs.codegen.target = "numpy"
#Initialization parameters are in octa_param 

class Octa(BuildingBlock):
    
    
    def __init__(self, name,
                 wtaParams = wtaParameters,
                 octaParams = octaParameters,           
                 neuron_eq_builder=octa_neuron,


                 monitor=True,
                 debug=False):


        self.num_input_neurons = octaParameters['num_input_neurons']
        self.num_neurons = octaParameters['num_neurons']
        
        BuildingBlock.__init__(self, name,
                               block_params,
                               debug,
                               monitor)


        self.sub_blocks, self._groups, self.monitors, self.standalone_params = gen_octa(name,
                                                              num_input_neurons=num_input_neurons,
                                                              num_neurons=num_neurons,
                                                               wtaParams = wtaParams,
                                                               octaParams = octaParams,
                                                               neuron_eq_builder= neuron_eq_builder,
                                                              monitor=monitor,
       
                                                              debug=debug,

                                                             )

        # Creating handles for neuron groups and inputs
        self.comp_wta = self.sub_blocks['comp_wta']
        self.pred_wta = self.sub_blocks['pred_wta']

        self.input_groups.update({'comp_n_spike_gen': self.comp_wta._groups['spike_gen']})
              
        self.output_groups.update({'comp_n_exc': self.comp_wta._groups['n_exc']})

        self.hidden_groups.update({
                'comp_n_inh' : compressionWTA._groups['n_inh'] ,
                'pred_n_exc' : predictionWTA._groups['n_exc'],
                'pred_n_inh' :  predictionWTA._groups['n_inh'],

    ##ADD TAGS to newly generated things
        error_connection._set_tags(tags_parameters.basic_tags_s_errror_con_octa, error_connection)


        }




def gen_octa(name, num_input_neurons, num_neurons, wtaParams, octaParams,\
             monitor, debug):
    
    """
        Generator function for the OCTA building block
    """

    if debug:
        print("Creating WTA's!")
    
    num_inh_neurons_c = int(num_neurons**2/4) 
    num_inh_neurons_p = int(num_input_neurons**2/4)
    
    compressionWTA = WTA(name='compressionWTA', dimensions=2, 
                         neuron_eq_builder=neuron_eq_builder,
                         num_neurons=num_neurons, num_inh_neurons=num_inh_neurons_c,
                         num_input_neurons=num_input_neurons, num_inputs=4, 
                         block_params=wtaParams,
                         monitor=monitor)
    
    predictionWTA = WTA(name='predictionWTA', dimensions=2,
                        neuron_eq_builder=neuron_eq_builder,
                        num_neurons=num_input_neurons, num_inh_neurons=num_inh_neurons_p,
                        num_input_neurons=num_neurons, num_inputs=4,
                        block_params=wtaParams, 
                        monitor=monitor)



    sub_blocks = {
            'compressionWTA' : compressionWTA,
            'predictionWTA' : predictionWTA,
            }
    
    
    #Replace spike_gen block with neurons obj instead of a spikegen
    replace_neurons(compressionWTA, 'spike_gen' ,num_input_neurons**2 ,\
                         equation_builder = neuron_eq_builder, refractory = wtaParams['rp_exc'])
    
    #Since the spikegen block has been changedd, the connections leading in/out of the group 
    #need to be updated
    replace_connection(compressionWTA, 'spike_gen', compressionWTA, 'n_exc',\
                       's_inp_exc', equation_builder= DPIstdp_gm)
    
    compressionWTA._set_tags(tags_parameters.basic_tags_s_inp_exc, compressionWTA._groups['s_inp_exc'])
    compressionWTA._groups['s_inp_exc'].weight =wtaParams['we_inp_exc']
    compressionWTA._groups['s_inp_exc'].taupre = octaParams['tau_stdp']
    compressionWTA._groups['s_inp_exc'].taupost = octaParams['tau_stdp']
    
   
    #Change the equation builder of the recurrent connections in compression WTA
    replace_connection(compressionWTA, 'n_exc', compressionWTA, 'n_exc',\
                       's_exc_exc', equation_builder= DPIstdp)
   
    compressionWTA._set_tags(tags_parameters.basic_tags_s_exc_exc, compressionWTA._groups['s_exc_exc'])
    compressionWTA._groups['s_exc_exc'].weight = wtaParams['we_exc_exc']

    #Changing the eqaution builder equation to include adaptation on the n_exc populatin
  
    replace_connection(compressionWTA, 'n_inh', compressionWTA, 'n_exc',\
                       's_inh_exc', equation_builder= DPIadp)
    compressionWTA._set_tags(tags_parameters.basic_tags_s_inh_exc, compressionWTA._groups['s_inh_exc'])
    compressionWTA._groups['s_inh_exc'].weight = octa_param.wtaParams['wi_inh_exc']
    compressionWTA._groups['s_inh_exc'].variance_th = np.random.uniform(low=octaParams['variance_th_c']-0.1,
                                                                 high=octaParams['variance_th_c']+0.1,
                                                                 size=len(compressionWTA._groups['s_inh_exc']))

    #Changing the eqaution builder equation to include adaptation on the n_exc population
    replace_connection(predictionWTA, 'n_inh', predictionWTA, 'n_exc',\
                       's_inh_exc', equation_builder= DPIadp)
    predictionWTA._set_tags(tags_parameters.basic_tags_s_inh_exc, predictionWTA._groups['s_inh_exc'])
    predictionWTA._groups['s_inh_exc'].weight = octa_param.wtaParams['wi_inh_exc']
    predictionWTA._groups['s_inh_exc'].variance_th = np.random.uniform(low=octaParams['variance_th_c']-0.1,
                                                                 high=octaParams['variance_th_c']+0.1,
                                                                 size=len(predictionWTA._groups['s_inh_exc']))

    #Modify the input of the prediction WTA. Bypassing the spike_gen block
    replace_connection(compressionWTA, 'n_exc', predictionWTA, 'n_exc',\
                       's_inp_exc', equation_builder= DPIadp)

    predictionWTA._set_tags(tags_parameters.basic_tags_s_inp_exc, predictionWTA._groups['s_inp_exc'])
    predictionWTA._groups['s_inp_exc']._tags['sign'] = 'exc'
    predictionWTA._groups['s_inp_exc']._tags['bb_type'] = 'octa'
    predictionWTA._groups['s_inp_exc']._tags['connection_type'] = 'ff'
    predictionWTA._groups['s_inp_exc'].weight = wtaParams['we_inp_exc']
    predictionWTA._groups['s_inp_exc'].taupre = octaParams['tau_stdp']
    predictionWTA._groups['s_inp_exc'].taupost = octaParams['tau_stdp']

    #Include stdp in recurrent connections in prediction WTA
    replace_connection(predictionWTA, 'n_exc', predictionWTA, 'n_exc',\
                       's_exc_exc', equation_builder= DPIstdp)
    compressionWTA._set_tags(tags_parameters.basic_tags_s_exc_exc, predictionWTA._groups['s_exc_exc'])
    predictionWTA._groups['s_exc_exc'].weight = wtaParams['we_exc_exc']

    #Set error and prediction connections
    error_connection, prediction_connection = error_prediction_connections(compressionWTA, predictionWTA)


    compressionWTA._groups['s_inh_exc'].inh_learning_rate = octaParams['inh_learning_rate']
    predictionWTA._groups['s_inh_exc'].inh_learning_rate = octaParams['inh_learning_rate']



    #Define all lists for initialiazations and initialiaze funcion
    w_init_group= [compressionWTA._groups['s_exc_exc']] +\
                [compressionWTA._groups['s_inp_exc']] +\ 
                [compressionWTA._groups['s_inh_exc']] + \
                [predictionWTA._groups['s_inh_exc']] +\
                [predictionWTA._groups['s_inp_exc']] +\
                [error_connection]
                
                
    
    add_weight_init(w_init_group, dist_param=octaParams['dist_param_init'], 
                                       scale=octaParams['scale_init'],
                                       distribution=octaParams['distribution'])
    weight_decay_group =  [compressionWTA._groups['s_inp_exc']] + \
                 [compressionWTA._groups['s_exc_exc']] +\
                [predictionWTA._groups['s_inp_exc']] + \
                [predictionWTA._groups['s_exc_exc']] + \
                [error_connection]

    weight_re_init_group= [compressionWTA._groups['s_inp_exc']] + \
                [compressionWTA._groups['s_exc_exc']] + \
                [predictionWTA._groups['s_inp_exc']] + \
                [predictionWTA._groups['s_exc_exc']] + \
                [error_connection]
                            
    pred_weight_decay_group = [prediction_connection]
    
    
    weight_regularization_group = [compressionWTA._groups['s_inp_exc']] + \
                                [error_connection]

    activity_proxy_group= [compressionWTA._groups['n_exc']] + [predictionWTA._groups['n_exc']]

    weight_re_init_ipred_group = [prediction_connection]


    #Add 
    add_decay_weight( weight_decay_group, octa_param.octaParams['weight_decay'], 
                 octa_param.octaParams['learning_rate'] )    

    add_weight_pred_decay( pred_weight_decay_group, octa_param.octaParams['weight_decay'],  octa_param.octaParams['learning_rate'] )    


    add_weight_re_init(weight_re_init_group,re_init_threshold=octaParams['re_init_threshold'],
                        dist_param_re_init=octaParams['dist_param_re_init'], 
                        scale_re_init=octaParams['scale_re_init'],
                        distribution=octaParams['distribution'])

    add_weight_re_init_ipred(weight_re_init_ipred_group, re_init_threshold=octa_param.octaParams['re_init_threshold'] )

    add_regulatization_weight(weight_regularization_group, buffer_size=octa_param.octaParams['buffer_size'] )

    add_proxy_activity(activity_proxy_group,  buffer_size=octa_param.octaParams['buffer_size_plast'],
                   decay=octa_param.octaParams['decay'])




    #initialiaze mismatch 
    add_bb_mismatch(compressionWTA)
    add_bb_mismatch(predictionWTA)
    error_connection.add_mismatch(mismatch_synap_param, seed= 42)










    
    def replace_neurons(bb, population, num_inputs, equation_builder, refractory):  
        
        bb._groups[population] = Neurons(num_inputs, equation_builder=equation_builder(num_inputs=3),
                                                        refractory=refractory,
                                                        name=bb._groups[population].name)
        
        
        bb._set_tags(tags_parameters.basic_tags_n_sg, bb._groups[population])
        bb._set_tags({'group_type' : 'Neuron'}, bb._groups[population])

        return None    


    def replace_connection(bb_source, population_source, bb_target, population_target , \
                           connection_name, equation_builder, method = 'euler'):   
        
        bb_target._groups[connection_name] = Connections(bb_source._groups[population_source],\
                                                 bb_target._groups[population_target],
                                                 equation_builder=equation_builder,
                                                  method=method,
                                                name=bb_target._groups[connection_name].name)
        bb_target._groups[connection_name].connect(True) 

        return None    
        
    
    
    
    def error_prediction_connections(compressionWTA, predictionWTA):
        
        #error connection between input neuron population and  prediction population
        error_connection = Connections(compressionWTA._groups['spike_gen'], 
                                   predictionWTA._groups['n_exc'],
                                   equation_builder=octa_param.DPIstdp_gm,
                                   method='euler', 
                                   name='error_connection')
        
        error_connection.connect('True')
        error_connection.weight = wtaParams['we_inp_exc']
        error_connection.taupre = octaParams['tau_stdp']
        error_connection.taupost = octaParams['tau_stdp']
        
        #prediction connection between prediction population and  input neuron population
    
        prediction_connection = Connections(predictionWTA._groups['n_exc'],
                                        compressionWTA._groups['spike_gen'],
                                        equation_builder=octa_param.SynSTDGM,
                                        method='euler',
                                        name='prediction_connection')
        prediction_connection.connect(True)
        prediction_connection.Ipred_plast = np.zeros((len(prediction_connection)))
    
        # Set learning rate
        compressionWTA._groups['s_inp_exc'].dApre = octaParams['learning_rate']
        compressionWTA._groups['s_exc_exc'].dApre = octaParams['learning_rate']
        predictionWTA._groups['s_inp_exc'].dApre = octaParams['learning_rate']
        predictionWTA._groups['s_exc_exc'].dApre = octaParams['learning_rate']
        
        error_connection.dApre = octaParams['learning_rate']
        prediction_connection.dApre = octaParams['learning_rate']
        
        return error_connection, prediction_connection
        
    
    
    
    
    
    
    