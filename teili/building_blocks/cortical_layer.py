# -*- coding: utf-8 -*-
""" This BuildingBlock class describes the organisation of a single 
cortical layer.
"""
# @Author: mmilde
# @Date:   2020-04-01 17:36:37

import sys
import time
import numpy as np

from brian2 import ms, mV, pA, SpikeGeneratorGroup,\
    SpikeMonitor, StateMonitor, core 

from teili.core.groups import Neurons, Connections
from teili.building_blocks.building_block import BuildingBlock, InhibitorySubnetwork
from teili.models.neuron_models import ExpAdaptLIF as neuron_model
from teili.models.synapse_models import AlphaStdp as synapse_model
from teili.models.synapse_models import Alpha as static_synapse_model

layer_params={
    'we_inp_exc': 1.5, 
    'we_exc_exc': 0.5,
    'we_exc_inh': 1, 
    'wi_inh_inh': -1,
    'wi_inh_exc': -1,
    'rp_exc': 1 * ms, 
    'rp_inh': 1 * ms,
    'num_neurons': 64, 
    'num_inh_neurons': None,
    'num_input_neurons': None, 
    'num_inputs': 3, 
    'num_inh_inputs': 3,
    'ee_connection_probability': 0.7,
    'ei_connection_probability': 0.7, 
    'ie_connection_probability': 0.7,
    'ii_connection_probability': 0.1,
    'connection_dropoff': 'exponential',
    }


class CorticalLayer(BuildingBlock, InhibitorySubnetwork):
    """ This class describes the connectivity between primary excitatory
    neurons, such as pyramidal cells, and a inhibitory sub-network consisting
    of models of PV- and SST-positive interneurons.
    """

    def __init__(self, 
                 name='layer*',
                 neuron_eq_builder=neuron_model,
                 synapse_eq_builder=synapse_model,
                 static_synapse_eq_builder=static_synapse_model,
                 block_params=layer_params,
                 monitor=True,
                 verbose=False):
        """ This function initialises the CorticalLayer building block.

        Args:
            name (str, optional): Name of the cortical layer
            neuron_eq_builder (class, optional): neuron class as imported from
                models/neuron_models.
            synapse_eq_builder (class, optional): synapse class as imported from
                models/synapse_models used for all synapses, unless 
                static_synapse_eq_builder is not `None`, in which case all
                non-plastic synapse follow the model provided.                
            static_synapse_eq_builder (class, optional): synapse class as imported from
                models/synapse_models for all non-plastic synapses.
            num_neurons (int, required): Number of excitatory neurons within
                the layer.
            num_inputs (int, optional): Number of input currents to WTA.
            spatial_kernel (str, optional): Connectivity kernel for lateral
                connectivity. Default is 'kernel_gauss_1d'.
                See tools.synaptic_kernel for more detail.
        """
        BuildingBlock.__init__(self,
                               name,
                               neuron_eq_builder,
                               synapse_eq_builder,
                               block_params,
                               verbose,
                               monitor)

        if static_synapse_eq_builder is None:
            static_synapse_eq_builder = synapse_eq_builder
        self._groups,\
        self.monitors,\
        self.standalone_parameters = gen_cortical_layer(name,
                                                        neuron_eq_builder,
                                                        synapse_eq_builder,
                                                        static_synapse_eq_builder,
                                                        **block_params)
        
        
        self.input_group
        self.output_group


def gen_cortical_layer(layer_name,
                       neuron_eq_builder=neuron_model,
                       synapse_eq_builder=synapse_model,
                       static_synapse_eq_builder=static_synapse_model,
                       we_inp_exc=1.5, 
                       we_exc_exc=0.5,
                       we_exc_inh=1, 
                       wi_inh_inh=-1,
                       wi_inh_exc=-1,
                       rp_exc=1 * ms, 
                       rp_inh=1 * ms,
                       num_neurons=64, 
                       num_inh_neurons=None,
                       num_input_neurons=None, 
                       num_inputs=3, 
                       num_inh_inputs=3,
                       ee_connection_probability=0.7,
                       ei_connection_probability=0.7, 
                       ie_connection_probability=0.7,
                       ii_connection_probability=0.1,
                       connection_dropoff='exponential',
                       additional_statevars=[], 
                       monitor=True, 
                       verbose=False):
    """ Combines InhibitorySubnetwork BuildingBloc with a primary excitatory
    neuron cell type such as pyramidal cells. 

    Args:
        layer_name (str, required): Name of the cortical layer.

    Returns:
        _groups (dictionary): Keys to all neuron and synapse groups.
        monitors (dictionary): Keys to all spike and state monitors.
        standalone_params (dictionary): Dictionary which holds all
            parameters to create a standalone network.

    """
    # create the input SpikeGeneratorGroup
    ts = np.asarray([]) * ms
    ind = np.asarray([])
    spike_gen = SpikeGeneratorGroup(
        num_input_neurons, indices=ind, times=ts,
        name=layer_name+ '_' + 'spike_gen')

    # create neuron groups
    n_exc = Neurons(num_neurons,
                    equation_builder=neuron_eq_builder(
                        num_inputs=3+num_inputs),
                    refractory=rp_exc,
                    name=layer_name + '__' + 'n_exc')

    s_exc_pv = Connections(n_exc,
                           interneurons._groups['n_pv'],
                           equation_builder=static_synapse_eq_builder
                           )

    s_exc_pv.connect(p=ei_connection_probability)
    

    Groups = {}
    Monitors = {}
    standalone_parameters = {}

    return Groups, Monitors, standalone_parameters
        
def set_layer_tags():
    pass