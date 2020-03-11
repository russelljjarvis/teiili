# -*- coding: utf-8 -*-
"""This module provides a reservoir network, as described in
Nicola and Clopath 2017.

Attributes:
    reservoir_params (dict): Dictionary of default parameters for reservoir.

Example:
    To use the Reservoir building block in your simulation you need
    to create an object of the class by:

    >>> from teili.building_blocks.reservoir import Reservoir
    >>> my_bb = Reservoir(name='my_reservoir')

    If you want to change the underlying neuron and synapse model you need to
    provide a different equation_builder class:

    >>> from teili.models.neuron_models import DPI as neuron_model
    >>> from teili.models.synapse_models import DPISyn as synapse_model
    >>> my_bb = Reservoir(name='my_reservoir',
                      neuron_eq_builder=DPI,
                      synapse_eq_builder=DPISyn)

    If you want to change the default parameters of your building block
    you need to define a dictionary, which you pass to the building_block

    >>> reservoir_params = {'weInpR': 1.5,
                            'weRInh': 1,
                            'wiInhR': -1,
                            'weRR': 0.5,
                            'sigm': 3,
                            'rpR': 0 * ms,
                            'rpInh': 0 * ms
                            }
    >>> my_bb = Reservoir(name='my_reservoir', block_params=reservoir_params)
"""
# @Author: ssolinas, mmilde, alpren
# @Date:   2017-12-27 10:46:44

import time
import sys
import numpy as np

from brian2 import ms, SpikeGeneratorGroup, SpikeMonitor,\
    StateMonitor, mV, pA, second

import teili.tools.synaptic_kernel

from teili.building_blocks.building_block import BuildingBlock
from teili.core.groups import Neurons, Connections

# from teili.models.neuron_models import DPI as neuron_model
# from teili.models.synapse_models import DPISyn as syn_model
# from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param
from teili.models.neuron_models import Izhikevich as neuron_model
from teili.models.synapse_models import DoubleExponential as synapse_model
from teili.models.parameters.izhikevich_param import parameters as neuron_model_params

reservoir_params = {'weInpR': 1.5,  # Excitatory synaptic weight
                    # between input SpikeGenerator and Reservoir neurons.
                    # Excitatory synaptic weight between Reservoir population
                    # and inhibitory interneuron.
                    'weRInh': 1,
                    # Inhibitory synaptic weight between inhibitory interneuron
                    # and Reservoir population.
                    'wiInhR': -1,
                    # Self-excitatory synaptic weight (within Reservoir).
                    'weRR': 0.5,
                    # Standard deviation in number of neurons for Gaussian
                    # connectivity kernel.
                    'sigm': 3,
                    'rpR': 0 * ms,  # Refractory period of Reservoir neurons.
                    'rpInh': 0 * ms  # Refractory period of inhibitory neurons.
                    }


class Reservoir(BuildingBlock):
    '''A recurrent Neural Net implementing a Reservoir.

    Attributes:
        group (dict): List of keys of neuron population.
        input_group (SpikeGenerator): SpikeGenerator object to stimulate Reservoir.
        num_neurons (int, optional): Size of Reservoir neuron population.
        fraction_inh_neurons (float, optional): Set to None to skip Dale's principle.
        spikemonR (brian2.SpikeMonitor object): A spikemonitor which monitors the activity of the
            reservoir population.
        standalone_params (dict): Keys for all standalone parameters necessary for cpp code generation.
    '''

    def __init__(self, name,
                 neuron_eq_builder=neuron_model,
                 synapse_eq_builder=synapse_model,
                 neuron_params=neuron_model_params,
                 block_params=reservoir_params,
                 num_neurons=16,
                 num_input_neurons=0,
                 num_output_neurons=1,
                 output_weights_init=[0],
                 Rconn_prob=None,
                 adjecency_mtr=None,
                 fraction_inh_neurons=0.2,
                 additional_statevars=[],
                 num_inputs=1,
                 spatial_kernel=None,
                 monitor=True,
                 verbose=False):
        """Summary

        Args:
            groupname (str, required): Name of the Reservoir population.
            dimensions (int, optional): Specifies if 1 or 2 dimensional Reservoir is created.
            neuron_eq_builder (class, optional): neuron class as imported from models/neuron_models.
            synapse_eq_builder (class, optional): synapse class as imported from models/synapse_models.
            block_params (dict, optional): Parameter for neuron populations.
            num_neurons (int, optional): Size of Reservoir neuron population.
            num_input_neurons (int, optional): Size of input population. If None, equal to size of Reservoir population
        adjecency_mtr (ndarray 3D, optional): 2D Connection matrix on the first layer of 3D, wieght matrix on the 2nd layer.
            fraction_inh_neurons (float, optional): Set to None to skip Dale's priciple.
            additional_statevars (list, optional): List of additional state variables which are not standard.
            num_inputs (int, optional): Number of input currents to Reservoir.
            monitor (bool, optional): Flag to auto-generate spike and state monitors.
            debug (bool, optional): Flag to gain additional information.

        Raises:
            NotImplementedError:
        """
        self.num_neurons = num_neurons
        BuildingBlock.__init__(self, name,
                               neuron_eq_builder,
                               synapse_eq_builder,
                               block_params,
                               verbose,
                               monitor)

        self.Groups, self.Monitors, \
            self.standalone_params = gen_reservoir(name,
                                                   neuron_eq_builder=neuron_eq_builder,
                                                   synapse_eq_builder=synapse_eq_builder,
                                                   neuron_model_params=neuron_model_params,
                                                   num_inputs=num_inputs,
                                                   Rconn_prob=Rconn_prob,
                                                   num_neurons=num_neurons,
                                                   num_input_neurons=num_input_neurons,
                                                   num_output_neurons=num_output_neurons,
                                                   output_weights_init=output_weights_init,
                                                   adjecency_mtr=adjecency_mtr,
                                                   fraction_inh_neurons=fraction_inh_neurons,
                                                   spatial_kernel=spatial_kernel,
                                                   monitor=monitor,
                                                   debug=verbose,
                                                   **block_params)

        if num_input_neurons:
            self.input_group = self.Groups['gRInpGroup']
        self.group = self.Groups['gRGroup']
        if monitor:
            self.spikemonR = self.Monitors['spikemonR']


def gen_reservoir(groupname,
                  neuron_eq_builder=neuron_model,
                  synapse_eq_builder=synapse_model,
                  neuron_model_params=neuron_model_params,
                  weInpR=1.5, weRInh=1, wiInhR=-1, weRR=0.5, sigm=3,
                  rpR=0 * ms, rpInh=0 * ms,
                  num_neurons=64,
                  Rconn_prob=None,
                  adjecency_mtr=None,
                  num_input_neurons=0,
                  num_output_neurons=1,
                  output_weights_init=[0],
                  taud=0,
                  taur=0,
                  num_inputs=1,
                  fraction_inh_neurons=0.2,
                  spatial_kernel="kernel_mexican_1d",
                  monitor=True, additional_statevars=[], debug=False):
    """Generates a reservoir network.

    Args:
        groupname (str, required): Name of the Reservoir population.
        neuron_eq_builder (class, optional): neuron class as imported from models/neuron_models.
        synapse_eq_builder (class, optional): synapse class as imported from models/synapse_models.
        weInpR (float, optional): Excitatory synaptic weight between input SpikeGenerator and Reservoir neurons.
        weRInh (int, optional): Excitatory synaptic weight between Reservoir population and inhibitory interneuron.
        wiInhR (TYPE, optional): Inhibitory synaptic weight between inhibitory interneuron and Reservoir population.
        weRR (float, optional): Self-excitatory synaptic weight (Reservoir).
        sigm (int, optional): Standard deviation in number of neurons for Gaussian connectivity kernel.
        rpR (float, optional): Refractory period of Reservoir neurons.
        rpInh (float, optional): Refractory period of inhibitory neurons.
        num_neurons (int, optional): Size of Reservoir neuron population.
        Rconn_prob (float, optional): Float 0<p<1 to set connection probability within the reservoir
        adjecency_mtr (numpy ndarray 3D of int and float, optional): Uses the adjecency matrix to set connections in the reservoir
        fraction_inh_neurons (int, optional): Set to None to skip Dale's principle.
        cutoff (int, optional): Radius of self-excitation.
        num_inputs (int, optional): Number of input currents to each neuron in the Reservoir.
        num_input_neurons (int, optional): Number of neurons in the input stage.
        num_readout_neurons (int, optional): Number of neurons in the readout stage.
        spatial_kernel (str, optional):  None defaults to kernel_mexican_1d
        monitor (bool, optional): Flag to auto-generate spike and statemonitors.
        additional_statevars (list, optional): List of additional state variables which are not standard.
        debug (bool, optional): Flag to gain additional information.

    Returns:
        Groups (dictionary): Keys to all neuron and synapse groups.
        Monitors (dictionary): Keys to all spike- and statemonitors.
        standalone_params (dictionary): Dictionary which holds all parameters to create a standalone network.
    """
    # time measurement
    start = time.clock()
    if spatial_kernel is None:
        spatial_kernel = "kernel_mexican_1d"

    spatial_kernel_func = getattr(
        teili.tools.synaptic_kernel, spatial_kernel)

    # create neuron groups
    gRGroup = Neurons(num_neurons,
                      equation_builder=neuron_eq_builder(
                          num_inputs=3 + num_inputs),
                      refractory=rpR, name='g' + groupname)
    # Initialize the neuronal voltage to the reset value
    # The set_params command is changing something else then just the parameters!!!
    # gRGroup.set_params(neuron_model_params)
    gRGroup.Vm = gRGroup.VR
    gRGroup.refP = rpR

    # create synapses
    synRR1e = Connections(gRGroup, gRGroup,
                          equation_builder=synapse_eq_builder(),
                          method="euler", name='s' + groupname + '_RR')
    # connect the nearest neighbors including itself
    if Rconn_prob:
        synRR1e.connect(p=Rconn_prob)
    # commenct according to the adjecency and weight matrix
    elif adjecency_mtr is not None:
        rows, cols = np.nonzero(adjecency_mtr[:, :, 0])
        synRR1e.connect(i=rows, j=cols)
        synRR1e.weight = adjecency_mtr[rows, cols, 1]
    else:
        print('Set either Rconn_prob or adjecency_mtr')
    # Initialize the time of last spike to a large number
    synRR1e.t_spike = 5000 * ms
    synRR1e.tausyne = taud
    synRR1e.tausyni = taud
    synRR1e.tausyne_rise = taur
    synRR1e.tausyni_rise = taur
    synRR1e.baseweight_e = 1. * pA
    synRR1e.baseweight_i = -1. * pA

    Groups = {'gRGroup': gRGroup,
              'synRR1e': synRR1e}

    if fraction_inh_neurons is not None:
        gRInhGroup = Neurons(round(num_neurons * fraction_inh_neurons),
                             equation_builder=neuron_eq_builder(num_inputs=1),
                             refractory=rpInh, name='g' + groupname + '_Inh')
        gRInhGroup.set_params(neuron_model_params)
        # create synapses
        synInhR1i = Connections(gRInhGroup, gRGroup,
                                equation_builder=synapse_eq_builder(),
                                method="euler", name='s' + groupname + '_Inhi')
        synRInh1e = Connections(gRGroup, gRInhGroup,
                                equation_builder=synapse_eq_builder(),
                                method="euler", name='s' + groupname + '_Inhe')
        synRInh1e.connect(p=1.0)  # Generates all to all connectivity
        synInhR1i.connect(p=1.0)
        synRInh1e.weight = weRInh
        synInhR1i.weight = wiInhR

        Groups.update({'gRInhGroup': gRInhGroup,
                       'synRInh1e': synRInh1e,
                       'synInhR1i': synInhR1i})

    # Set input for Reservoir group
    if num_input_neurons > 0:
        # Create the input layer
        ts_reservoir = np.asarray([]) * ms
        ind_reservoir = np.asarray([])
        gRInpGroup = SpikeGeneratorGroup(
            num_input_neurons, indices=ind_reservoir,
            times=ts_reservoir, name='g' + groupname + '_Inp')

        # create synapses
        synInpR1e = Connections(
            gRInpGroup, gRGroup,
            equation_builder=synapse_eq_builder(),
            method="euler", name='s' + groupname + '_Inp')

        # connect synapses
        synInpR1e.connect(p=1.0)
        # set weights
        synInpR1e.weight = 0

        Groups.update({'gRInpGroup': gRInpGroup,
                       'synInpR1e': synInpR1e})

    # Set output readout layer for Reservoir group
    if num_output_neurons > 0:
        # Create a simple integrator neuron
        simple_integrator = 'rate : 1'
        simple_integrator_on_pre = '''h += 1 /(taur * taud / ms)'''
        # Create the output layer
        gROutGroup = Neurons(
            num_output_neurons,
            model=simple_integrator,
            name='g' + groupname + '_Out',
            parameters='')

        # create readout synapses
        synOutR1e = Connections(gRGroup, gROutGroup,
                                model="""dr/dt = -r/taud + h : 1 (clock-driven)
                                dh/dt = -h/taur : second **-1 (clock-driven)
                                rate_post =  weight * r : 1 (summed)
                                taud = %f * ms : second
                                taur = %f * ms : second
                                weight : 1
                                """ % (taud / ms, taur / ms),
                                on_pre=simple_integrator_on_pre,
                                method="euler",
                                name='s' + groupname + '_Out',
                                parameters='')
        # connect synapses

        synOutR1e.connect()
        # set weights
        synOutR1e.weight = output_weights_init.reshape(synOutR1e.weight.shape)

        Groups.update({'gROutGroup': gROutGroup,
                       'synOutR1e': synOutR1e})

    # spikemons
    if monitor:
        spikemonR = SpikeMonitor(gRGroup, name='spikemon' + groupname + '_R')
        Monitors = {'spikemonR': spikemonR}
        if fraction_inh_neurons is not None:
            spikemonRInh = SpikeMonitor(
                gRInhGroup, name='spikemon' + groupname + '_RInh')
            Monitors['spikemonRInh'] = spikemonRInh
        if num_input_neurons > 0:
            Spikemonrinp = SpikeMonitor(
                gRInpGroup, name='spikemon' + groupname + '_RInp')
            Monitors['spikemonRInp'] = spikemonRInp
        try:
            statemonR = StateMonitor(gRGroup, ('Vm', 'Iexp', 'Iin', 'Iconst', 'Ie0', 'Ii0'), record=True,
                                     name='statemon' + groupname + '_R')
        except KeyError:
            statemonR = StateMonitor(gRGroup, ('Iexp', 'Iin'), record=True,
                                     name='statemon' + groupname + '_R')
        Monitors['statemonR'] = statemonR
        # Monitors['statemon_neurons_rate'] = StateMonitor(gRateOutGroup, ('rate'), record=True,
        # name='statemon' + groupname + '_neurons_rate')
        if num_output_neurons > 0:
            Monitors['statemon_readout_rate'] = StateMonitor(gROutGroup, ('rate'), record=True,
                                                             name='statemon' + groupname + '_readout_rate')
            Monitors['statemon_neuron_rates'] = StateMonitor(synOutR1e,
                                                             {'r'},
                                                             record=True,
                                                             name='statemonRout')

    # replacevars should be the 'real' names of the parameters, that can be
    # changed by the arguments of this function:
    # in this case: weInpR, weRInh, wiInhR, weRR,rpR, rpInh,sigm
    standalone_params = {
        # synRR1e.name + '_latWeight': weRR,
        # synRR1e.name + '_latSigma': sigm,
        gRGroup.name + '_refP': rpR
    }
    if fraction_inh_neurons is not None:
        standalone_params[synRInh1e.name + '_weight'] = weRInh
        standalone_params[synInhR1i.name + '_weight'] = wiInhR
        standalone_params[gRInhGroup.name + '_refP'] = rpInh,
    if num_input_neurons > 0:
        standalone_params[synInpR1e.name + '_weight'] = weInpR

    end = time.clock()
    if debug:
        print('creating Reservoir of ' + str(num_neurons) + ' neurons with name ' +
              groupname + ' took ' + str(end - start) + ' sec')
        print('The keys of the output dict are:')
        for key in Groups:
            print(key)

    return Groups, Monitors, standalone_params
