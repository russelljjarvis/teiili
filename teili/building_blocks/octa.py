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
from brian2 import ms
from brian2 import  SpikeGeneratorGroup
from brian2 import prefs

import matplotlib.pyplot as plt
import numpy as np
import time

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
DPIadp, DPIstdp_gm ,SynSTDGM,mismatch_synap_param

import teili.tools.tags_parameters as tags_parameters
from teili.tools.octa_tools.octa_tools import add_decay_weight, add_weight_re_init, add_weight_re_init_ipred,\
add_proxy_activity, add_regulatization_weight, add_weight_pred_decay, add_bb_mismatch, add_weight_init
from teili.tools.octa_tools.lock_and_load import  save_monitor, load_monitor,\
save_weights, load_weights

prefs.codegen.target = "numpy"
#Initialization parameters are located in octa_param


class Octa(BuildingBlock):
    
    
    def __init__(self, name,
                 wtaParams = wtaParameters,
                 octaParams = octaParameters,           
                 neuron_eq_builder=octa_neuron,
                 stacked_inp = True,
                noise= True,
                 monitor=True,
                 debug=False):


        self.num_input_neurons = octaParameters['num_input_neurons']
        self.num_neurons = octaParameters['num_neurons']
        block_params = {}
        block_params.update(wtaParams)
        block_params.update(octaParams)

        BuildingBlock.__init__(self, name,
                               neuron_eq_builder,
                               block_params,
                               debug,
                               monitor)


        self.sub_blocks, self._groups, self.monitors, self.standalone_params = gen_octa(name,
                                                              num_input_neurons=self.num_input_neurons,
                                                              num_neurons=self.num_neurons,
                                                               wtaParams = wtaParams,
                                                               octaParams = octaParams,
                                                               neuron_eq_builder= neuron_eq_builder,
                                                               stacked_inp=noise,
                                                                noise  = noise,
                                                              monitor=monitor,
                                                              debug=debug,
                                                             )

#         Creating handles for neuron groups and inputs
        self.comp_wta = self.sub_blocks['compressionWTA']
        self.pred_wta = self.sub_blocks['predictionWTA']

        self.input_groups.update({'comp_n_spike_gen': self.comp_wta._groups['spike_gen']})
        self.output_groups.update({'comp_n_exc': self.comp_wta._groups['n_exc']})

        self.hidden_groups.update({
                'comp_n_inh' : self.comp_wta._groups['n_inh'] ,
                'pred_n_exc' : self.pred_wta._groups['n_exc'],
                'pred_n_inh' :  self.pred_wta._groups['n_inh']})

#        self._groups['error_connection']._set_tags(tags_parameters.basic_tags_compression_con_octa, 
 #                   self._groups['error_connection'])
  #      self._groups['prediction_connection']._set_tags(tags_parameters.basic_tags_prediction_con_octa, 
   #                 self._groups['prediction_connection'])

def gen_octa(name, num_input_neurons, num_neurons, wtaParams, octaParams, neuron_eq_builder,
             stacked_inp= True, noise = True, monitor= True, debug = True):
    """
        Generator function for the OCTA building block
    """

    if debug:
        print("Creating WTA's!")

    #Timing    
    start = time.time()

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

    compressionWTA._set_tags(tags_parameters.basic_wta_n_sg, compressionWTA._groups['spike_gen'])
    compressionWTA._set_tags({'group_type' : 'Neuron'}, compressionWTA._groups['spike_gen'])
    
    #Since the spikegen block has been changedd, the connections leading in/out of the group 
    #need to be updated
    replace_connection(compressionWTA, 'spike_gen', compressionWTA, 'n_exc',\
                       's_inp_exc', equation_builder= DPIstdp_gm)
    
    compressionWTA._set_tags(tags_parameters.basic_wta_s_inp_exc, compressionWTA._groups['s_inp_exc'])
    compressionWTA._groups['s_inp_exc'].weight =wtaParams['we_inp_exc']
    compressionWTA._groups['s_inp_exc'].taupre = octaParams['tau_stdp']
    compressionWTA._groups['s_inp_exc'].taupost = octaParams['tau_stdp']
    
   
    #Change the equation builder of the recurrent connections in compression WTA
    replace_connection(compressionWTA, 'n_exc', compressionWTA, 'n_exc',\
                       's_exc_exc', equation_builder= DPIstdp)
   
    compressionWTA._set_tags(tags_parameters.basic_wta_s_exc_exc, compressionWTA._groups['s_exc_exc'])
    compressionWTA._groups['s_exc_exc'].weight = wtaParams['we_exc_exc']

    #Changing the eqaution builder equation to include adaptation on the n_exc populatin
  
    replace_connection(compressionWTA, 'n_inh', compressionWTA, 'n_exc',\
                       's_inh_exc', equation_builder= DPIadp)
    compressionWTA._set_tags(tags_parameters.basic_wta_s_inh_exc, compressionWTA._groups['s_inh_exc'])
    compressionWTA._groups['s_inh_exc'].weight = wtaParams['wi_inh_exc']
    compressionWTA._groups['s_inh_exc'].variance_th = np.random.uniform(low=octaParams['variance_th_c']-0.1,
                                                                 high=octaParams['variance_th_c']+0.1,
                                                                 size=len(compressionWTA._groups['s_inh_exc']))

    #Changing the eqaution builder equation to include adaptation on the n_exc population
    replace_connection(predictionWTA, 'n_inh', predictionWTA, 'n_exc',\
                       's_inh_exc', equation_builder= DPIadp)
    predictionWTA._set_tags(tags_parameters.basic_wta_s_inh_exc, predictionWTA._groups['s_inh_exc'])
    predictionWTA._groups['s_inh_exc'].weight = wtaParams['wi_inh_exc']
    predictionWTA._groups['s_inh_exc'].variance_th = np.random.uniform(low=octaParams['variance_th_c']-0.1,
                                                                 high=octaParams['variance_th_c']+0.1,
                                                                 size=len(predictionWTA._groups['s_inh_exc']))

    #Modify the input of the prediction WTA. Bypassing the spike_gen block
    replace_connection(compressionWTA, 'n_exc', predictionWTA, 'n_exc',\
                       's_inp_exc', equation_builder= DPIstdp)

    predictionWTA._set_tags(tags_parameters.basic_wta_s_inp_exc, predictionWTA._groups['s_inp_exc'])
    predictionWTA._groups['s_inp_exc']._tags['sign'] = 'exc'
    predictionWTA._groups['s_inp_exc']._tags['bb_type'] = 'octa'
    predictionWTA._groups['s_inp_exc']._tags['connection_type'] = 'ff'
    predictionWTA._groups['s_inp_exc'].weight = wtaParams['we_inp_exc']
    predictionWTA._groups['s_inp_exc'].taupre = octaParams['tau_stdp']
    predictionWTA._groups['s_inp_exc'].taupost = octaParams['tau_stdp']

    #Include stdp in recurrent connections in prediction WTA
    replace_connection(predictionWTA, 'n_exc', predictionWTA, 'n_exc',\
                       's_exc_exc', equation_builder= DPIstdp)
    compressionWTA._set_tags(tags_parameters.basic_wta_s_exc_exc, predictionWTA._groups['s_exc_exc'])
    predictionWTA._groups['s_exc_exc'].weight = wtaParams['we_exc_exc']

    #Set error and prediction connections
    error_connection, prediction_connection = error_prediction_connections(compressionWTA,
                                                                           predictionWTA,
                                                                           wtaParams,
                                                                           octaParams)


    compressionWTA._groups['s_inh_exc'].inh_learning_rate = octaParams['inh_learning_rate']
    predictionWTA._groups['s_inh_exc'].inh_learning_rate = octaParams['inh_learning_rate']

    #Define all lists for initialiazations and initialiaze function
#     todo:           [compressionWTA._groups['s_inh_exc']]+\ 
    
#Important check s_inh_exc exists in the generated connectons
    
    w_init_group= [compressionWTA._groups['s_exc_exc']]+\
                [compressionWTA._groups['s_inp_exc']]+\
                [predictionWTA._groups['s_inh_exc']]+\
                [predictionWTA._groups['s_inp_exc']]+\
                [error_connection]
                
    add_weight_init(w_init_group, dist_param=octaParams['dist_param_init'], 
                                       scale=octaParams['scale_init'],
                                       distribution=octaParams['distribution'])

    weight_decay_group =  [compressionWTA._groups['s_inp_exc']] + \
                 [compressionWTA._groups['s_exc_exc']] +\
                [predictionWTA._groups['s_inp_exc']] + \
                [predictionWTA._groups['s_exc_exc']] + \
                [error_connection]
    add_decay_weight( weight_decay_group, decay_strategy =octaParams['weight_decay'], 
                 learning_rate=octaParams['learning_rate'] )  

    weight_re_init_group= [compressionWTA._groups['s_inp_exc']] + \
                [compressionWTA._groups['s_exc_exc']] + \
                [predictionWTA._groups['s_inp_exc']] + \
                [predictionWTA._groups['s_exc_exc']] + \
                [error_connection]
    add_weight_re_init(weight_re_init_group,re_init_threshold=octaParams['re_init_threshold'],
                        dist_param_re_init=octaParams['dist_param_re_init'], 
                        scale_re_init=octaParams['scale_re_init'],
                        distribution=octaParams['distribution'])
                            
    pred_weight_decay_group = [prediction_connection]
    add_weight_pred_decay( pred_weight_decay_group, decay_strategy=octaParams['weight_decay'],  
                        learning_rate=octaParams['learning_rate'] )    

    weight_regularization_group = [compressionWTA._groups['s_inp_exc']] + \
                                [error_connection]
    add_regulatization_weight(weight_regularization_group, buffer_size=octaParams['buffer_size'] )


    activity_proxy_group= [compressionWTA._groups['n_exc']] + [predictionWTA._groups['n_exc']]
    add_proxy_activity(activity_proxy_group,  buffer_size=octaParams['buffer_size_plast'],
                   decay=octaParams['decay'])

    weight_re_init_ipred_group = [prediction_connection]
    add_weight_re_init_ipred(weight_re_init_ipred_group, re_init_threshold=octaParams['re_init_threshold'] )


    #initialiaze mismatch 
    add_bb_mismatch(compressionWTA)
    add_bb_mismatch(predictionWTA)
    error_connection.add_mismatch(mismatch_synap_param, seed= 42)


    if stacked_inp: 
        inputGroup = SpikeGeneratorGroup(N=num_input_neurons**2, indices=[], times=[]*ms)
    
        inputSynapse = Connections(inputGroup, compressionWTA._groups['spike_gen'],
                                   equation_builder=DPISyn(),
                                   method='euler',
                                   name='inputSynapse')
    
        inputSynapse.connect('i==j')
        inputSynapse.weight = 3250.
        testbench_stim = OCTA_Testbench()
    
        testbench_stim.rotating_bar(length=10, nrows=10, 
                                direction='cw', 
                                ts_offset=3, angle_step=10, 
                                noise_probability=0.2, repetitions=octaParams['revolutions'], debug=False)
        
        inputGroup.set_spikes(indices=testbench_stim.indices, times=testbench_stim.times * ms)
    


    if noise:
        testbench_c = WTA_Testbench()
        testbench_p = WTA_Testbench()
    
    
        testbench_c.background_noise(num_neurons=num_neurons, rate=10)
        testbench_p.background_noise(num_neurons=num_input_neurons, rate=10)
    
        noise_syn_c_exc = Connections(testbench_c.noise_input, 
                                compressionWTA._groups['n_exc'],
                                equation_builder=DPISyn(), 
                                name="noise_syn_c_exc")
        #
        noise_syn_c_exc.connect("i==j")
        noise_syn_c_exc.weight = octaParams['noise_weight']
    
        noise_syn_p_exc = Connections(testbench_c.noise_input, 
                                predictionWTA._groups['n_exc'],
                                equation_builder=DPISyn(), 
                                name="noise_syn_p_exc")
    
    
    
        noise_syn_p_exc.connect("i==j")
        noise_syn_p_exc.weight = octaParams['noise_weight']
    synGroups =   {     
#            'comp_s_inp_exc':  compressionWTA._groups['s_inp_exc'],
#            'comp_s_exc_exc': compressionWTA._groups['s_exc_exc'],
#            'comp_s_exc_inh': compressionWTA._groups['s_exc_inh'],
#            'comp_s_inh_exc': compressionWTA._groups['s_inh_exc'],
#        
#            'pred_s_inp_exc': predictionWTA._groups['s_inp_exc'],
#            'pred_s_exc_exc': predictionWTA._groups['s_exc_exc'],
#            'pred_s_exc_inh': predictionWTA._groups['s_exc_inh'],
#            'pred_s_inh_exc':  predictionWTA._groups['s_inh_exc'],
            'error_connection': error_connection,
            'prediction_connection' : prediction_connection      
        }

    if stacked_inp:
        input_dict = { 'inputSynapse': inputSynapse }   
        synGroups.update(input_dict)   
    
    
    neurGroups ={
#        'comp_n_exc' : compressionWTA._groups['n_exc'],
#        'comp_n_inh' : compressionWTA._groups['n_inh'] ,
#        'comp_n_spike_gen' :  compressionWTA._groups['spike_gen'],
#        'pred_n_exc' : predictionWTA._groups['n_exc'],
#        'pred_n_inh' :  predictionWTA._groups['n_inh'],
        }
    if stacked_inp:
        input_sync= {'inputGroup': inputGroup}   

        neurGroups.update(input_sync)   
    if noise:
        noise_syn= {'pred_noise_syn_exc' : noise_syn_p_exc,
                    'comp_noise_syn_exc' : noise_syn_c_exc,
                    'pred_noise_gen' : testbench_p.noise_input,
                    'comp_noise_gen' : testbench_c.noise_input
                }
        neurGroups.update(noise_syn)  
    
    monitors = {}
    if monitors:
        monitors = {

         'comp_spikemon_exc' : compressionWTA.monitors['spikemon_exc'], 
         'comp_spikemon_inh' :compressionWTA.monitors['spikemon_inh'],
         'comp_spikemon_inp' :compressionWTA.monitors['spikemon_inp'],                 
         'comp_statemon_exc' :compressionWTA.monitors['statemon_exc'],
         
         'pred_spikemon_exc' : predictionWTA.monitors['spikemon_exc'], 
         'pred_spikemon_inh' :predictionWTA.monitors['spikemon_inh'],
         'pred_statemon_exc' :predictionWTA.monitors['statemon_exc'], 
         
         }
    #We are not creating a monitor on the stimulus because should be in spikemon_inp
    group= {}
    group.update(synGroups)
    group.update(neurGroups)
    
    end = time.time()
    
    if debug:
            print('Creating octa ' +  ' took ' + str(end - start) + ' sec!')
            print('The keys of the output dict are:')
            for key in group:
                print(key)
                
    standalone_params = {}
    
    
    return sub_blocks, group, monitors, standalone_params


def replace_neurons(bb, population, num_inputs, equation_builder, refractory):  
    
    bb._groups[population] = Neurons(num_inputs, equation_builder=equation_builder(num_inputs=3),
                                                    refractory=refractory,
                                                    name=bb._groups[population].name)
    

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
    



def error_prediction_connections(compressionWTA, predictionWTA,wtaParams, octaParams):
    
    #error connection between input neuron population and  prediction population
    error_connection = Connections(compressionWTA._groups['spike_gen'], 
                               predictionWTA._groups['n_exc'],
                               equation_builder=DPIstdp_gm,
                               method='euler', 
                               name='error_connection')
    
    error_connection.connect('True')
    error_connection.weight = wtaParams['we_inp_exc']
    error_connection.taupre = octaParams['tau_stdp']
    error_connection.taupost = octaParams['tau_stdp']
    
    #prediction connection between prediction population and  input neuron population

    prediction_connection = Connections(predictionWTA._groups['n_exc'],
                                    compressionWTA._groups['spike_gen'],
                                    equation_builder=SynSTDGM,
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
    



if __name__ == '__main__':
    test_OCTA =  Octa(name='test_OCTA', 
                wtaParams = wtaParameters,
                 octaParams = octaParameters,     
                 neuron_eq_builder=octa_neuron,
                 stacked_inp = True,
                noise= True,
                 monitor=True,
                 debug=True)

    test_OCTA_2 =  Octa(name='test_OCTA_2', 
                wtaParams = wtaParameters,
                 octaParams = octaParameters,     
                 neuron_eq_builder=octa_neuron,
                 stacked_inp = True,
                noise= True,
                 monitor=True,
                 debug=True)
    Net = teiliNetwork()
    Net.add(test_OCTA, test_OCTA_2)
   # Net.run(octaParameters['duration'] * ms, report='text')
    
    