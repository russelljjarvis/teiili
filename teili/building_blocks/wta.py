# -*- coding: utf-8 -*-
"""This module provides different Winner-Takes_all (WTA) circuits.

Beside different dimensionality of the WTA, i.e 1D & 2D, you can select
different spatial connectivity and neuron and synapse models.

Attributes:
    wta_params (dict): Dictionary of default parameters for wta.

Todo:
    * Generalize for n dimensions

Example:
    To use the WTA building block in your simulation you need
    to create an object of the class by doing:

    >>> from teili.building_blocks.wta import WTA
    >>> my_bb = WTA(name='my_wta')

    If you want to change the underlying neuron and synapse model you need to
    provide a different equation_builder class:

    >>> from teili.models.neuron_models import ExpAdaptIF
    >>> from teili.models.synapse_models import ReversalSynV
    >>> my_bb = WTA(name='my_wta',
                      neuron_eq_builder=ExpAdaptIF,
                      synapse_eq_builder=ReversalSynV)

    If you want to change the default parameters of your building block
    you need to define a dictionary, which you pass to the building_block:

    >>> wta_params = {'weInpWTA': 1.5,
                      'weWTAInh': 1,
                      'wiInhWTA': -1,
                      'weWTAWTA': 0.5,
                      'sigm': 3,
                      'rpWTA': 3 * ms,
                      'rpInh': 1 * ms,
                      'EI_connection_probability': 1,
                      'IE_connection_probability': 1,
                      'II_connection_probability': 0
                      }
    >>> my_bb = WTA(name='my_wta', block_params=wta_params)
"""
# @Author: mmilde, alpren
# @Date:   2017-12-27 10:46:44


import time
import numpy as np
import sys

# import brian2
from brian2 import ms, mV, pA, SpikeGeneratorGroup, SpikeMonitor, StateMonitor

import teili.tools.synaptic_kernel
from teili.tools.misc import print_states
from teili.tools.distance import dist1d2dint
from teili.tools.indexing import ind2x, ind2y
# from teili.tools.plotting import plot_spikemon_qt, plot_statemon_qt

from teili.building_blocks.building_block import BuildingBlock
from teili.core.groups import Neurons, Connections

from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn

wta_params = {'weInpWTA': 1.5,
              'weWTAInh': 1,
              'wiInhWTA': -1,
              'weWTAWTA': 0.5,
              'sigm': 3,
              'rpWTA': 3 * ms,
              'rpInh': 1 * ms,
              'EI_connection_probability': 1,
              'IE_connection_probability': 1,
              'II_connection_probability': 0
              }


class WTA(BuildingBlock):
    '''A 1 or 2D square Winner-Takes_all (WTA) Building block.

    Attributes:
        dimensions (int, optional): Specifies if 1 or 2 dimensional WTA is
            created.
        group (dict): List of keys of neuron population.
        inhGroup (TYPE): Description
        inputGroup (brian2.SpikeGenerator obj.): SpikeGenerator object to
             stimulate WTA.
        num_neurons (int, optional): Size of WTA neuron population.
        plot_win (TYPE): Description
        spikemonWTA (brian2.SpikeMonitor obj.): A spike monitor which monitors
            the activity of the WTA population.

    Deleted Attributes:
        standalone_params (dict): Keys for all standalone parameters necessary
            for cpp code generation.
    '''

    def __init__(self, name='wta*',
                 dimensions=1,
                 neuron_eq_builder=DPI,
                 synapse_eq_builder=DPISyn,
                 block_params=wta_params,
                 num_neurons=16,
                 num_inh_neurons=2,
                 num_input_neurons=None,
                 cutoff=10,
                 additional_statevars=[],
                 num_inputs=1,
                 spatial_kernel=None,
                 monitor=True,
                 debug=False):
        """Initializes building block object with defined dimensionality and
        connectivity scheme.

        Args:
            name (str): Name of the WTA BuildingBlock
            dimensions (int, optional): Specifies if 1 or 2 dimensional WTA is
                created.
            neuron_eq_builder (class, optional): neuron class as imported from
                models/neuron_models.
            synapse_eq_builder (class, optional): synapse class as imported from
                models/synapse_models.
            block_params (dict, optional): Parameter for neuron populations.
            num_neurons (int, optional): Size of WTA neuron population.
            num_inh_neurons (int, optional): Size of inhibitory interneuron
                population.
            num_input_neurons (int, optional): Size of input population.
                If None, equal to size of WTA population.
            cutoff (int, optional): Radius of self-excitation.
            additional_statevars (list, optional): List of additonal state
                variables which are not standard.
            num_inputs (int, optional): Number of input currents to WTA.
            spatial_kernel (None, optional): Description
            monitor (bool, optional): Flag to auto-generate spike and state
                monitors.
            debug (bool, optional): Flag to gain additional information.

        Raises:
            NotImplementedError: If dimensions is not 1 or 2.

        Deleted Parameters:
            groupname (str, required): Name of the WTA population.
        """
        self.num_neurons = num_neurons
        self.dimensions = dimensions
        BuildingBlock.__init__(self,
                               name,
                               neuron_eq_builder,
                               synapse_eq_builder,
                               block_params,
                               debug,
                               monitor)

        if dimensions == 1:
            self._groups,\
            self.monitors,\
            self.standalone_params = gen1dWTA(name,
                                              neuron_eq_builder,
                                              synapse_eq_builder,
                                              num_neurons=num_neurons,
                                              num_inh_neurons=num_inh_neurons,
                                              additional_statevars=additional_statevars,
                                              cutoff=cutoff,
                                              num_input_neurons=num_input_neurons,
                                              num_inputs=num_inputs,
                                              monitor=monitor,
                                              debug=debug,
                                              spatial_kernel=spatial_kernel,
                                              **block_params)
        elif dimensions == 2:
            self._groups,\
            self.monitors,\
            self.standalone_params = gen2dWTA(name,
                                              neuron_eq_builder,
                                              synapse_eq_builder,
                                              num_neurons=num_neurons,
                                              num_inh_neurons=num_inh_neurons,
                                              additional_statevars=additional_statevars,
                                              cutoff=cutoff,
                                              num_input_neurons=num_input_neurons,
                                              num_inputs=num_inputs,
                                              monitor=monitor,
                                              debug=debug,
                                              spatial_kernel=spatial_kernel,
                                              **block_params)

        else:
            raise NotImplementedError("only 1 and 2 d WTA available, sorry")

        BuildingBlock.input = self._groups['gWTAInpGroup']
        BuildingBlock.output =
        BuildingBlock.hidden = self._groups['']
        self.inhGroup = self._groups['gWTAInhGroup']
        self._group = self._groups['gWTAGroup']

        if monitor:
            self.spikemonWTA = self.monitors['spikemonWTA']


def gen1dWTA(groupname,
             neuron_eq_builder=DPI,
             synapse_eq_builder=DPISyn,
             weInpWTA=1.5, weWTAInh=1, wiInhWTA=-1, weWTAWTA=0.5, sigm=3,
             rpWTA=3 * ms, rpInh=1 * ms,
             num_neurons=64, num_inh_neurons=5, num_input_neurons=None, cutoff=10, num_inputs=1,
             spatial_kernel="kernel_gauss_1d",
             EI_connection_probability=1, IE_connection_probability=1, II_connection_probability=0,
             monitor=True, additional_statevars=[], debug=False):
    """Creates a 1D WTA population of neurons, including the inhibitory interneuron population

    Args:
        groupname (str, required): Name of the WTA population.
        neuron_eq_builder (class, optional): neuron class as imported from models/neuron_models.
        synapse_eq_builder (class, optional): synapse class as imported from models/synapse_models.
        weInpWTA (float, optional): Excitatory synaptic weight between input SpikeGenerator and WTA neurons.
        weWTAInh (int, optional): Excitatory synaptic weight between WTA population and inhibitory interneuron.
        wiInhWTA (TYPE, optional): Inhibitory synaptic weight between inhibitory interneuron and WTA population.
        weWTAWTA (float, optional): Self-excitatory synaptic weight (WTA).
        sigm (int, optional): Standard deviation in number of neurons for Gaussian connectivity kernel.
        rpWTA (float, optional): Refractory period of WTA neurons.
        rpInh (float, optional): Refractory period of inhibitory neurons.
        num_neurons (int, optional): Size of WTA neuron population.
        num_inh_neurons (int, optional): Size of inhibitory interneuron population.
        num_input_neurons (int, optional): Size of input population. If None, equal to size of WTA population.
        cutoff (int, optional): Radius of self-excitation.
        num_inputs (int, optional): Number of input currents to WTA.
        spatial_kernel (str, optional): Description
        EI_connection_probability (float, optional): WTA to interneuron connectivity probability.
        IE_connection_probability (float, optional): Interneuron to WTA connectivity probability
        II_connection_probability (float, optional): Interneuron to Interneuron connectivity probability.
        monitor (bool, optional): Flag to auto-generate spike and state monitors.
        additional_statevars (list, optional): List of additional state variables which are not standard.
        debug (bool, optional): Flag to gain additional information.

    Returns:
        _groups (dictionary): Keys to all neuron and synapse groups.
        Monitors (dictionary): Keys to all spike and state monitors.
        standalone_params (dictionary): Dictionary which holds all parameters to create a standalone network.
    """
    if spatial_kernel is None:
        spatial_kernel = "kernel_gauss_1d"

    if type(spatial_kernel) == brian2.core.functions.Function:
        spatial_kernel_func = spatial_kernel
        spatial_kernel_name = spatial_kernel.pyfunc.__name__
    else:
        spatial_kernel_func = getattr(
            teili.tools.synaptic_kernel, spatial_kernel)
        spatial_kernel_name = spatial_kernel

    # time measurement
    start = time.time()

    # create neuron groups
    gWTAGroup = Neurons(num_neurons, equation_builder=neuron_eq_builder(num_inputs=3 + num_inputs),
                        refractory=rpWTA, name='g' + groupname)
    gWTAInhGroup = Neurons(num_inh_neurons, equation_builder=neuron_eq_builder(num_inputs=2),
                           refractory=rpInh, name='g' + groupname + '_Inh')

    if num_input_neurons is None:
        num_input_neurons = num_neurons
    # empty input for WTA group
    tsWTA = np.asarray([]) * ms
    indWTA = np.asarray([])
    gWTAInpGroup = SpikeGeneratorGroup(
        num_input_neurons, indices=indWTA, times=tsWTA, name='g' + groupname + '_Inp')

    # create synapses
    synInpWTA1e = Connections(gWTAInpGroup, gWTAGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inpe')
    synWTAWTA1e = Connections(gWTAGroup, gWTAGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_e')
    synInhWTA1i = Connections(gWTAInhGroup, gWTAGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inhi')
    synWTAInh1e = Connections(gWTAGroup, gWTAInhGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inhe')
    synInhInh1i = Connections(gWTAInhGroup, gWTAInhGroup,
                              equation_builder=synapse_eq_builder(),
                              method='euler', name='s' + groupname + '_i')

    # connect synapses
    synInpWTA1e.connect('i==j')
    # connect the nearest neighbors including itself
    synWTAWTA1e.connect('abs(i-j)<=cutoff')
    # Generates all to all connectivity
    synWTAInh1e.connect('True', p=EI_connection_probability)
    synInhWTA1i.connect('True', p=IE_connection_probability)
    synInhInh1i.connect('True', p=II_connection_probability)

    synWTAWTA1e.add_state_variable(
        name='latWeight', shared=True, constant=True)
    synWTAWTA1e.add_state_variable(name='latSigma', shared=True, constant=True)

    # set weights
    synInpWTA1e.weight = weInpWTA
    synWTAInh1e.weight = weWTAInh
    synInhWTA1i.weight = wiInhWTA
    # lateral excitation kernel
    # we add an additional attribute to that synapse, which allows us to change
    # and retrieve that value more easily
    synWTAWTA1e.latWeight = weWTAWTA
    synWTAWTA1e.latSigma = sigm
    synWTAWTA1e.namespace.update({spatial_kernel_name: spatial_kernel_func})
    synWTAWTA1e.weight = 'latWeight * ' + \
        spatial_kernel_name + '(i,j,latSigma)'

    _groups = {
        'gWTAGroup': gWTAGroup,
        'gWTAInhGroup': gWTAInhGroup,
        'gWTAInpGroup': gWTAInpGroup,
        'synInpWTA1e': synInpWTA1e,
        'synWTAWTA1e': synWTAWTA1e,
        'synWTAInh1e': synWTAInh1e,
        'synInhWTA1i': synInhWTA1i}

    # spikemons
    if monitor:
        spikemonWTA = SpikeMonitor(
            gWTAGroup, name='spikemon' + groupname + '_WTA')
        spikemonWTAInh = SpikeMonitor(
            gWTAInhGroup, name='spikemon' + groupname + '_WTAInh')
        spikemonWTAInp = SpikeMonitor(
            gWTAInpGroup, name='spikemon' + groupname + '_WTAInp')
        try:
            statemonWTA = StateMonitor(gWTAGroup, ('Vm', 'Ie', 'Ii'), record=True,
                                       name='statemon' + groupname + '_WTA')
        except KeyError:
            statemonWTA = StateMonitor(gWTAGroup, ('Imem', 'Iin'), record=True,
                                       name='statemon' + groupname + '_WTA')
        monitors = {
            'spikemonWTA': spikemonWTA,
            'spikemonWTAInh': spikemonWTAInh,
            'spikemonWTAInp': spikemonWTAInp,
            'statemonWTA': statemonWTA}

    # replacevars should be the 'real' names of the parameters, that can be
    # changed by the arguments of this function:
    # in this case: weInpWTA, weWTAInh, wiInhWTA, weWTAWTA,rpWTA, rpInh,sigm
    standalone_params = {
        synInpWTA1e.name + '_weight': weInpWTA,
        synWTAInh1e.name + '_weight': weWTAInh,
        synInhWTA1i.name + '_weight': wiInhWTA,
        synWTAWTA1e.name + '_latWeight': weWTAWTA,
        synWTAWTA1e.name + '_latSigma': sigm,
        gWTAGroup.name + '_refP': rpWTA,
        gWTAInhGroup.name + '_refP': rpInh,
    }

    end = time.time()
    if debug:
        print('creating WTA of ' + str(num_neurons) + ' neurons with name ' +
              groupname + ' took ' + str(end - start) + ' sec')
        print('The keys of the output dict are:')
        for key in _groups:
            print(key)

    return _groups, monitors, standalone_params


def gen2dWTA(groupname,
             neuron_eq_builder=DPI,
             synapse_eq_builder=DPISyn,
             weInpWTA=1.5, weWTAInh=1, wiInhWTA=-1, weWTAWTA=2, sigm=2.5,
             rpWTA=2.5 * ms, rpInh=1 * ms,
             wiInhInh=0, EI_connection_probability=1., IE_connection_probability=1.,
             II_connection_probability=0.1,
             spatial_kernel="kernel_gauss_2d",
             num_neurons=20, num_inh_neurons=3, num_input_neurons=None, cutoff=9, num_inputs=1,
             monitor=True, additional_statevars=[], debug=False):
    '''Creates a 2D square WTA population of neurons, including the inhibitory interneuron population

    Args:
        groupname (str, required): Name of the WTA population.
        neuron_eq_builder (class, optional): neuron class as imported from models/neuron_models.
        synapse_eq_builder (class, optional): synapse class as imported from models/synapse_models.
        weInpWTA (float, optional): Excitatory synaptic weight between input SpikeGenerator and WTA neurons.
        weWTAInh (int, optional): Excitatory synaptic weight between WTA population and inhibitory interneuron.
        wiInhWTA (TYPE, optional): Inhibitory synaptic weight between inhibitory interneuron and WTA population.
        weWTAWTA (float, optional): Self-excitatory synaptic weight (WTA).
        sigm (int, optional): Standard deviation in number of neurons for Gaussian connectivity kernel.
        rpWTA (float, optional): Refractory period of WTA neurons.
        rpInh (float, optional): Refractory period of inhibitory neurons.
        wiInhInh (int, optional): Self-inhibitory weight of the interneuron population.
        EI_connection_probability (float, optional): WTA to interneuron connectivity probability.
        IE_connection_probability (float, optional): Interneuron to WTA connectivity probability
        II_connection_probability (float, optional): Interneuron to Interneuron connectivity probability.
        spatial_kernel (str, optional): Description
        num_neurons (int, optional): Size of WTA neuron population.
        num_inh_neurons (int, optional): Size of inhibitory interneuron population.
        num_input_neurons (int, optional): Size of input population. If None, equal to size of WTA population.
        cutoff (int, optional): Radius of self-excitation.
        num_inputs (int, optional): Number of input currents to WTA.
        monitor (bool, optional): Flag to auto-generate spike and statemonitors.
        additional_statevars (list, optional): List of additional state variables which are not standard.
        debug (bool, optional): Flag to gain additional information.

    Returns:
        _groups (dictionary): Keys to all neuron and synapse groups.
        Monitors (dictionary): Keys to all spike and state monitors.
        standalone_params (dictionary): Dictionary which holds all parameters to create a standalone network.
    '''

    if spatial_kernel is None:
        spatial_kernel = "kernel_gauss_2d"

    if type(spatial_kernel) == brian2.core.functions.Function:
        spatial_kernel_func = spatial_kernel
        spatial_kernel_name = spatial_kernel.pyfunc.__name__
    else:
        spatial_kernel_func = getattr(
            teili.tools.synaptic_kernel, spatial_kernel)
        spatial_kernel_name = spatial_kernel
    # time measurement
    start = time.time()

    # create neuron groups
    num2dNeurons = num_neurons**2
    num_inh_inputs = 2
    gWTAGroup = Neurons(num2dNeurons, equation_builder=neuron_eq_builder(num_inputs=3 + num_inputs),
                        refractory=rpWTA, name='g' + groupname)
    gWTAInhGroup = Neurons(num_inh_neurons, equation_builder=neuron_eq_builder(num_inputs=num_inh_inputs),
                           refractory=rpInh, name='g' + groupname + '_Inh')

    gWTAGroup.namespace['num_neurons'] = num_neurons
    gWTAGroup.namespace['ind2x'] = ind2x
    gWTAGroup.namespace['ind2y'] = ind2y
    gWTAGroup.x = "ind2x(i, num_neurons,num_neurons)"
    gWTAGroup.y = "ind2y(i, num_neurons,num_neurons)"

    if num_input_neurons is None:
        num_input2d_neurons = num2dNeurons
    else:
        num_input2d_neurons = num_input_neurons**2
    # empty input for WTA group
    tsWTA = np.asarray([]) * ms
    indWTA = np.asarray([])
    gWTAInpGroup = SpikeGeneratorGroup(
        num_input2d_neurons, indices=indWTA, times=tsWTA, name='g' + groupname + '_Inp')

    # create synapses
    synInpWTA1e = Connections(gWTAInpGroup, gWTAGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inpe')
    synWTAWTA1e = Connections(gWTAGroup, gWTAGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_e')
    synInhWTA1i = Connections(gWTAInhGroup, gWTAGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inhi')
    synWTAInh1e = Connections(gWTAGroup, gWTAInhGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inhe')
    synInhInh1i = Connections(gWTAInhGroup, gWTAInhGroup,
                              equation_builder=synapse_eq_builder(),
                              method='euler', name='s' + groupname + '_i')

    # connect synapses
    synInpWTA1e.connect('i==j')
    # connect the nearest neighbors including itself
    synWTAWTA1e.connect('dist1d2dint(i,j,num_neurons,num_neurons)<=cutoff')
    # Generates all to all connectivity
    synWTAInh1e.connect('True', p=EI_connection_probability)
    synInhWTA1i.connect('True', p=IE_connection_probability)
    synInhInh1i.connect('True', p=II_connection_probability)

    synWTAWTA1e.add_state_variable(
        name='latWeight', shared=True, constant=True)
    synWTAWTA1e.add_state_variable(name='latSigma', shared=True, constant=True)

    # set weights
    synInpWTA1e.weight = weInpWTA
    synWTAInh1e.weight = weWTAInh
    synInhWTA1i.weight = wiInhWTA
    synInhInh1i.weight = wiInhInh

    # lateral excitation kernel
    # we add an additional attribute to that synapse, which allows us to change
    # and retrieve that value more easily
    synWTAWTA1e.latWeight = weWTAWTA
    synWTAWTA1e.latSigma = sigm
    synWTAWTA1e.namespace[spatial_kernel_name] = spatial_kernel_func
    synWTAWTA1e.namespace['num_neurons'] = num_neurons
    synWTAWTA1e.weight = 'latWeight * ' + spatial_kernel_name + \
        '(i,j,latSigma,num_neurons,num_neurons)'

    _groups = {
        'gWTAGroup': gWTAGroup,
        'gWTAInhGroup': gWTAInhGroup,
        'gWTAInpGroup': gWTAInpGroup,
        'synInpWTA1e': synInpWTA1e,
        'synWTAWTA1e': synWTAWTA1e,
        'synWTAInh1e': synWTAInh1e,
        'synInhWTA1i': synInhWTA1i,
        'synInhInh1i': synInhInh1i}

    # spikemons
    spikemonWTA = SpikeMonitor(gWTAGroup, name='spikemon' + groupname + '_WTA')
    spikemonWTAInh = SpikeMonitor(
        gWTAInhGroup, name='spikemon' + groupname + '_WTAInh')
    spikemonWTAInp = SpikeMonitor(
        gWTAInpGroup, name='spikemon' + groupname + '_WTAInp')
    try:
        statemonWTA = StateMonitor(gWTAGroup, ('Vm', 'Ie', 'Ii'), record=True,
                                   name='statemon' + groupname + '_WTA')
    except KeyError:
        statemonWTA = StateMonitor(gWTAGroup, ('Imem', 'Iin'), record=True,
                                   name='statemon' + groupname + '_WTA')
    monitors = {
        'spikemonWTA': spikemonWTA,
        'spikemonWTAInh': spikemonWTAInh,
        'spikemonWTAInp': spikemonWTAInp,
        'statemonWTA': statemonWTA}

    # replacevars should be the real names of the parameters,
    # that can be changed by the arguments of this function:
    # in this case: weInpWTA, weWTAInh, wiInhWTA, weWTAWTA,rpWTA, rpInh,sigm
    standalone_params = {
        synInpWTA1e.name + '_weight': weInpWTA,
        synWTAInh1e.name + '_weight': weWTAInh,
        synInhWTA1i.name + '_weight': wiInhWTA,
        synInhInh1i.name + '_weight': wiInhInh,
        synWTAWTA1e.name + '_latWeight': weWTAWTA,
        synWTAWTA1e.name + '_latSigma': sigm,
        gWTAGroup.name + '_refP': rpWTA,
        gWTAInhGroup.name + '_refP': rpInh,
    }

    end = time.time()
    if debug:
        print('creating WTA of ' + str(num_neurons) + ' x ' + str(num_neurons) + ' neurons with name ' +
              groupname + ' took ' + str(end - start) + ' sec')
        print('The keys of the output dict are:')
        for key in _groups:
            print(key)

    return _groups, monitors, standalone_params
