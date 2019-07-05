#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:58:50 2019

@author: matteo


This module provides a hierarchical building block called OCTA.
            -Online Clustering of Temporal Activity-
            
This is a level 2 hierarchical building block, it uses basic building blocks such
as the WTA.

If you want to change the default parameters of your building block
    you need to define a dictionary, which you pass to the building_block:




"""
from brian2 import ms
from brian2 import  SpikeGeneratorGroup, SpikeMonitor
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
DPIadp ,SynSTDGM,mismatch_synap_param

import teili.tools.tags_parameters as tags_parameters
from teili.tools.octa_tools.octa_tools import add_decay_weight, add_weight_re_init, add_weight_re_init_ipred,\
add_proxy_activity, add_regulatization_weight, add_weight_pred_decay, add_bb_mismatch, add_weight_init
from teili.tools.octa_tools.lock_and_load import  save_monitor, load_monitor,\
save_weights, load_weights

from teili.tools.visualizer.DataControllers import Rasterplot
from teili.tools.visualizer.DataViewers import PlotSettings

from teili.tools.octa_tools.weight_init import weight_init 

prefs.codegen.target = "numpy"


class Octa(BuildingBlock):
    
    
    def __init__(self, name,
                 neuron_eq_builder=octa_neuron,
                 wtaParams = wtaParameters,
                 octaParams = octaParameters,   
                 num_input_neurons= 10,
                 num_neurons = 7,
                 stacked_inp = True,
                noise= True,
                 monitor=True,
                 debug=False):
        """Initializes building block object with defined
        connectivity scheme.

        Args:
            name (str): Name of the OCTA BuildingBlock
 
            neuron_eq_builder (class, optional): neuron class as imported from
                models/neuron_models.
            wtaParams (Dict, optional): WTA dictionary
            octaParams (Dict, optional): octa parameter dictionary
            num_neurons (int, optional): Size of WTA neuron population.
            num_input_neurons (int, optional): Size of input population.
            stacked_inp (bool, optional): Flag to include an input in the form of a rotating bar
            noise (bool, optional): Flag to include a noise
            monitor (bool, optional): Flag to auto-generate spike and state
                monitors.
            debug (bool, optional): Flag to gain additional information.
        
        """

        self.num_input_neurons = num_input_neurons
        self.num_neurons =num_neurons
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
                                                                noise = noise,
                                                              monitor=monitor,
                                                              debug=debug,
                                                             )

#        Creating handles for neuron groups and inputs
        self.comp_wta = self.sub_blocks['compressionWTA']
        self.pred_wta = self.sub_blocks['predictionWTA']

#       Ddfining input - output groups
        
        self.input_groups.update({'comp_n_spike_gen': self.comp_wta._groups['spike_gen']})
        self.output_groups.update({'comp_n_exc': self.comp_wta._groups['n_exc']})

        self.hidden_groups.update({
                'comp_n_inh' : self.comp_wta._groups['n_inh'] ,
                'pred_n_exc' : self.pred_wta._groups['n_exc'],
                'pred_n_inh' :  self.pred_wta._groups['n_inh']})
    
        set_OCTA_tags(self,  self._groups)
def gen_octa(groupname, num_input_neurons, num_neurons, wtaParams, octaParams, neuron_eq_builder,
             stacked_inp= True, noise = True, monitor= True, debug = False):
    """
        Generator function for the OCTA building block
 Args:
            name (str): Name of the OCTA BuildingBlock
 
            neuron_eq_builder (class, optional): neuron class as imported from
                models/neuron_models.
            wtaParams (Dict, optional): WTA dictionary
            octaParams (Dict, optional): octa parameter dictionary
            num_neurons (int, optional): Size of WTA neuron population.
            num_input_neurons (int, optional): Size of input population.
            stacked_inp (bool, optional): Flag to include an input in the form of a rotating bar
            noise (bool, optional): Flag to include a noise
            monitor (bool, optional): Flag to auto-generate spike and state
                monitors.
            debug (bool, optional): Flag to gain additional information.

    Returns:
        sub_blocks (dictionary): Keys to all sub_blocks of the network
        _groups (dictionary): Keys to all neuron and synapse groups specific to this BB.
        monitors (dictionary): Keys to all spike and state monitors specific to this BB.

    Functional information about the network:
        
        The OCTA network is an implementation of the canonical microcircuit found in the 
        cortex levareging temporal information to extract meaning from the input data. 
        It consists of two WTA networks (compression and prediction) connected in a recurrent
        manner (figure of connectivity can be found in the docs). 
        Every building block in the teili implementation has a cortical counterpart for which
        the connectivity and function is preserved:
            compressionWTA['spike_gen'] : Layer 2/3
            compressionWTA['n_exc'] : Layer 4
            predictionWTA['n_exc'] : Layer 6

    Given a high dimensional input in L2/3 the network extracts in the recurrent connections of
    L4 a lower dimensional representation of temporal dependencies by learniing spatio-temporal features.
    
    
    """
    if debug:
        print("Creating WTA's!")
        print("Stacked Input: ", stacked_inp)
        print("noise",noise)
        print("monitor " ,monitor)
        

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

    if debug:
        print("Created compressionWTA and predictionWTA")
    #Replace spike_gen block with neurons obj instead of a spikegen
    replace_neurons(compressionWTA, 'spike_gen' ,num_input_neurons**2 ,\
                         equation_builder = neuron_eq_builder, refractory = wtaParams['rp_exc'])
    compressionWTA._set_tags(tags_parameters.basic_wta_n_sg, compressionWTA._groups['spike_gen'])
    compressionWTA._set_tags({'group_type' : 'Neuron'}, compressionWTA._groups['spike_gen'])
    
    #Since the spikegen block has been changedd, the connections leading in/out of the group 
    #need to be updated
    replace_connection(compressionWTA, 'spike_gen', compressionWTA, 'n_exc',\
                       's_inp_exc', equation_builder= DPIstdp)
    
    
    compressionWTA._set_tags(tags_parameters.basic_wta_s_inp_exc, compressionWTA._groups['s_inp_exc'])
    compressionWTA._groups['s_inp_exc'].weight =wtaParams['we_inp_exc']
    compressionWTA._groups['s_inp_exc'].taupre = octaParams['tau_stdp']
    compressionWTA._groups['s_inp_exc'].taupost = octaParams['tau_stdp']
    
  
    #Change the equation builder of the recurrent connections in compression WTA
    replace_connection(compressionWTA, 'n_exc', compressionWTA, 'n_exc',\
                       's_exc_exc', equation_builder= DPIstdp, name = 'compressionWTA'+
                       '_n_exc_exc')
   
    compressionWTA._set_tags(tags_parameters.basic_wta_s_exc_exc, compressionWTA._groups['s_exc_exc'])
    compressionWTA._groups['s_exc_exc'].weight = wtaParams['we_exc_exc']

    #Changing the eqaution builder equation to include adaptation on the n_exc populatiOn
  
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
    predictionWTA._groups['s_inh_exc'].variance_th = np.random.uniform(low=octaParams['variance_th_p']-0.1,
                                                                 high=octaParams['variance_th_p']+0.1,
                                                                 size=len(predictionWTA._groups['s_inh_exc']))
  

#    #Modify the input of the prediction WTA. Bypassing the spike_gen block
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


# lists with the groups to be added to specific functions and run_regular
    
    w_init_group= [compressionWTA._groups['s_exc_exc']]+\
                [compressionWTA._groups['s_inp_exc']]+\
                [compressionWTA._groups['s_inh_exc']]+\
                [predictionWTA._groups['s_inh_exc']]+\
                [predictionWTA._groups['s_inp_exc']]+\
                [predictionWTA._groups['s_exc_exc']] +\
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
                                   name=groupname + '_inputSynapse')
    
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
                                name= groupname + '_noise_comp_exc')
        #
        noise_syn_c_exc.connect("i==j")
        noise_syn_c_exc.weight = octaParams['noise_weight']
    
        noise_syn_p_exc = Connections(testbench_c.noise_input, 
                                predictionWTA._groups['n_exc'],
                                equation_builder=DPISyn(), 
                                name=groupname + '_noise_pred_exc')
    
    
    
        noise_syn_p_exc.connect("i==j")
        noise_syn_p_exc.weight = octaParams['noise_weight']
        compressionWTA._groups['n_exc']._tags['noise'] = 1
        predictionWTA._groups['n_exc']._tags['noise'] = 1

    synGroups =   {     
           
            'error_connection': error_connection,
            'prediction_connection' : prediction_connection      
        }

    if stacked_inp:
        input_dict = { 'inputSynapse': inputSynapse }   
        synGroups.update(input_dict)   
    
    
    neurGroups ={     }
    
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
    if monitor:
        compressionWTA.monitors['spikemon_inp'] = SpikeMonitor(compressionWTA._groups['spike_gen'], 
                      name='spikemon_inp')
        
        monitors = {
                'comp_spikemon_inp' : compressionWTA.monitors['spikemon_inp'],                 
         
         }
        if stacked_inp:
            stacked_inp_monitors = {
                    'input' : SpikeMonitor( inputGroup, name=groupname + '_inputGroup')
             
             }
            monitors.update(stacked_inp_monitors)
            
    group= {}
    group.update(synGroups)
    group.update(neurGroups)
    
    end = time.time()
    
    if debug:
            print('Creating octa ' +  ' took ' + str(end - start) + ' sec!')
            print('The keys of the ' + groupname + ' output dict are:')
            for key in group:
                print(key)
                
    standalone_params = {}
    
    
    sub_blocks = {
            'compressionWTA' : compressionWTA,
            'predictionWTA' : predictionWTA,
            }
    
    octaParameters['duration'] = np.max(testbench_stim.times)

    
    return sub_blocks, group, monitors, standalone_params


def replace_neurons(bb, population, num_inputs, equation_builder, refractory): 
    '''
    This function replaces the neuron population inside a building block with a different 
    equation rule
    
    '''
    
    bb._groups[population] = Neurons(num_inputs, equation_builder=equation_builder(num_inputs=4),
                                                    refractory=refractory,
                                                    name=bb._groups[population].name)
    

    return None    


def replace_connection(bb_source, population_source, bb_target, population_target , \
                       connection_name, equation_builder, method = 'euler', name  = None):   
    '''
    This function replaces the connection between two populations with a different 
    equation rule
    
    '''
    
    if name == None:
        name = bb_target._groups[connection_name].name
        
    bb_target._groups[connection_name] = Connections(bb_source._groups[population_source],\
                                             bb_target._groups[population_target],
                                             equation_builder=equation_builder,
                                              method=method,
                                            name=name)
    bb_target._groups[connection_name].connect(True) 

    return None    
    


def error_prediction_connections(compressionWTA, predictionWTA,wtaParams, octaParams):
    '''
    Set connection between the spike_gen of compression layer and n_exc of prediction layer
    '''    
    error_connection = Connections(compressionWTA._groups['spike_gen'], 
                               predictionWTA._groups['n_exc'],
                               equation_builder=DPIstdp,
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


def set_OCTA_tags(self, _groups):
    '''
    Sets default tags to a OCTA network

'''    
    self._set_tags(tags_parameters.basic_octa_error_connection, _groups['error_connection'])
    self._set_tags(tags_parameters.basic_octa_prediction_connection, _groups['prediction_connection'])
    self._set_tags(tags_parameters.basic_octa_inputSyn, _groups['inputSynapse'])
    self._set_tags(tags_parameters.basic_octa_inpGroup, _groups['inputGroup'])
    self._set_tags(tags_parameters.basic_octa_pred_noise_syn_exc, _groups['pred_noise_syn_exc'])
    self._set_tags(tags_parameters.basic_octa_comp_noise_syn_exc, _groups['comp_noise_syn_exc'])
    self._set_tags(tags_parameters.basic_octa_pred_noise_gen, _groups['pred_noise_gen'])
    self._set_tags(tags_parameters.basic_octa_comp_noise_gen, _groups['comp_noise_gen'])



def plot_sorted_compression(OCTA):
    
    '''
    Plot the spiking activity of the compression layer sorted by similarity index.
    
     The out-of-the-box network the spiking activity should align with the input timing
    
    '''

    weights = OCTA.sub_blocks['compressionWTA'].groups['s_exc_exc'].w_plast
    s = SortMatrix(nrows=49, ncols=49, matrix=weights, axis=1)
    monitor = OCTA.sub_blocks['compressionWTA'].monitors['spikemon_exc']
    moni = np.asarray([np.where(np.asarray(s.permutation) == int(i))[0][0] for i in monitor.i])
    plt.figure(1)
    plt.plot(monitor.t, moni , '.r')
    plt.xlabel("Time")
    plt.ylabel("Sorted spikes")
    plt.title("Rasterplot compression block")

if __name__ == '__main__':
    
    Net = teiliNetwork()


    test_OCTA =  Octa(name='test_OCTA', 
                wtaParams = wtaParameters,
                 octaParams = octaParameters,     
                 neuron_eq_builder=octa_neuron,
                 num_input_neurons= octaParameters['num_input_neurons'],
                 num_neurons = octaParameters['num_neurons'],
                 stacked_inp = True,
                 noise= True,
                 monitor=True,
                 debug=False)

    Net.add(      
            test_OCTA,
            test_OCTA.sub_blocks['predictionWTA'],
            test_OCTA.sub_blocks['compressionWTA']
          )

    Net.run(octaParameters['duration']*ms, report='text')
    
    plot_sorted_compression(OCTA=test_OCTA)
 

    