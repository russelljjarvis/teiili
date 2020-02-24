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

    >>> wta_params = {'we_inp_exc': 1.5,
                    'we_exc_inh': 1,
                    'wi_inh_exc': -1,
                    'we_exc_exc': 0.5,
                    'wi_inh_inh': -1,
                    'sigm': 3,
                    'rp_exc': 3 * ms,
                    'rp_inh': 1 * ms,
                    'ei_connection_probability': 1,
                    'ie_connection_probability': 1,
                    'ii_connection_probability': 0
                    }
        >>> my_bb = WTA(name='my_wta', block_params=wta_params)
    """
# @Author: mmilde, alpren
# @Date:   2017-12-27 10:46:44

import sys
import time
import numpy as np

from brian2 import ms, mV, pA, SpikeGeneratorGroup,\
    SpikeMonitor, StateMonitor, core

import teili.tools.synaptic_kernel
from teili.tools.misc import print_states
from teili.tools.distance import dist1d2dint
from teili.tools.indexing import ind2x, ind2y

from teili.building_blocks.building_block import BuildingBlock
from teili.core.groups import Neurons, Connections

from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn

from teili.core import tags as tags_parameters


wta_params = {'we_inp_exc': 1.5,
              'we_exc_inh': 1,
              'wi_inh_exc': -1,
              'we_exc_exc': 0.5,
              'wi_inh_inh': -1,
              'sigm': 3,
              'rp_exc': 3 * ms,
              'rp_inh': 1 * ms,
              'ei_connection_probability': 1,
              'ie_connection_probability': 1,
              'ii_connection_probability': 0
              }


class WTA(BuildingBlock):
    '''A 1 or 2D square Winner-Takes_all (WTA) Building block.

    Attributes:
        dimensions (int, optional): Specifies if 1 or 2 dimensional WTA is
            created.
        num_neurons (int, optional): Size of WTA neuron population.
        spike_gen (brian2.Spikegenerator obj.): SpikeGenerator group of the BB
        spikemon_exc (brian2.SpikeMonitor obj.): A spike monitor which monitors
            the activity of the WTA population.
        standalone_params (dict): Dictionary of parameters to be changed after
            standalone code generation.
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
                 verbose=False):
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
            spatial_kernel (str, optional): Connectivity kernel for lateral
                connectivity. Default is 'kernel_gauss_1d'.
                See tools.synaptic_kernel for more detail.
            monitor (bool, optional): Flag to auto-generate spike and state
                monitors.
            verbose (bool, optional): Flag to gain additional information.

        Raises:
            NotImplementedError: If dimensions is not 1 or 2.
        """
        self.num_neurons = num_neurons
        self.dimensions = dimensions
        BuildingBlock.__init__(self,
                               name,
                               neuron_eq_builder,
                               synapse_eq_builder,
                               block_params,
                               verbose,
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
                                                  verbose=verbose,
                                                  spatial_kernel=spatial_kernel,
                                                  **block_params)
            set_wta_tags(self, self._groups)

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
                                                  verbose=verbose,
                                                  spatial_kernel=spatial_kernel,
                                                  **block_params)

            set_wta_tags(self, self._groups)

        else:
            raise NotImplementedError(
                "only 1 and 2 d WTA available, sorry")
        self.spike_gen = self._groups['spike_gen']
        self.input_groups.update({'n_exc': self._groups['n_exc']})
        self.output_groups.update({'n_exc': self._groups['n_exc']})
        self.hidden_groups.update({'n_inh': self._groups['n_inh']})

        if monitor:
            self.spikemon_exc = self.monitors['spikemon_exc']


def gen1dWTA(groupname,
             neuron_eq_builder=DPI,
             synapse_eq_builder=DPISyn,
             we_inp_exc=1.5, we_exc_inh=1, wi_inh_inh=-1,
             wi_inh_exc=-1, we_exc_exc=0.5,
             sigm=3, rp_exc=3 * ms, rp_inh=1 * ms,
             num_neurons=64, num_inh_neurons=5,
             num_input_neurons=None, num_inputs=1, num_inh_inputs=2,
             cutoff=10, spatial_kernel="kernel_gauss_1d",
             ei_connection_probability=1, ie_connection_probability=1,
             ii_connection_probability=0,
             additional_statevars=[], monitor=True, verbose=False):
    """Creates a 1D WTA population of neurons, including the inhibitory
    interneuron population

    Args:
        groupname (str, required): Name of the WTA population.
        neuron_eq_builder (class, optional): neuron class as imported from
            models/neuron_models.
        synapse_eq_builder (class, optional): synapse class as imported from
            models/synapse_models.
        we_inp_exc (float, optional): Excitatory synaptic weight between
            input SpikeGenerator and WTA neurons.
        we_exc_inh (float, optional): Excitatory synaptic weight between WTA
            population and inhibitory interneuron.
        wi_inh_inh (float, optional): Inhibitory synaptic weight between
            interneurons.
        wi_inh_exc (float, optional): Inhibitory synaptic weight between
            inhibitory interneuron and WTA population.
        we_exc_exc (float, optional): Self-excitatory synaptic weight (WTA).
        sigm (int, optional): Standard deviation in number of neurons for
            Gaussian connectivity kernel.
        rp_exc (float, optional): Refractory period of WTA neurons.
        rp_inh (float, optional): Refractory period of inhibitory neurons.
        num_neurons (int, optional): Size of WTA neuron population.
        num_inh_neurons (int, optional): Size of inhibitory interneuron
            population.
        num_input_neurons (int, optional): Size of input population.
            If None, equal to size of WTA population.
        num_inputs (int, optional): Number of input currents to WTA.
        num_inh_inputs (int, optional): Number of input currents to the
            inhibitory group.
        cutoff (int, optional): Radius of self-excitation.
        spatial_kernel (str, optional): Connectivity kernel for lateral
            connectivity. Default is 'kernel_gauss_1d'.
            See tools.synaptic_kernel for more detail.
        ei_connection_probability (float, optional): WTA to interneuron
            connectivity probability.
        ie_connection_probability (float, optional): Interneuron to WTA
            connectivity probability
        ii_connection_probability (float, optional): Interneuron to
            Interneuron connectivity probability.
        additional_statevars (list, optional): List of additional state
            variables which are not standard.
        monitor (bool, optional): Flag to auto-generate spike and state
            monitors.
        verbose (bool, optional): Flag to gain additional information.

    Returns:
        _groups (dictionary): Keys to all neuron and synapse groups.
        monitors (dictionary): Keys to all spike and state monitors.
        standalone_params (dictionary): Dictionary which holds all
            parameters to create a standalone network.
    """
    if spatial_kernel is None:
        spatial_kernel = "kernel_gauss_1d"

    if type(spatial_kernel) == core.functions.Function:
        spatial_kernel_func = spatial_kernel
        spatial_kernel_name = spatial_kernel.pyfunc.__name__
    else:
        spatial_kernel_func = getattr(
            teili.tools.synaptic_kernel, spatial_kernel)
        spatial_kernel_name = spatial_kernel

    # time measurement
    start = time.time()

    # create neuron groups
    n_exc = Neurons(num_neurons,
                    equation_builder=neuron_eq_builder(
                        num_inputs=3+num_inputs),
                    refractory=rp_exc,
                    name=groupname + '__' + 'n_exc')
    n_inh = Neurons(num_inh_neurons,
                    equation_builder=neuron_eq_builder(
                        num_inputs=num_inh_inputs),
                    refractory=rp_inh,
                    name=groupname + '__' + 'n_inh')

    if num_input_neurons is None:
        num_input_neurons = num_neurons
    # empty input for WTA group
    ts = np.asarray([]) * ms
    ind = np.asarray([])
    spike_gen = SpikeGeneratorGroup(
        num_input_neurons, indices=ind, times=ts,
        name=groupname + '_' + 'spike_gen')

    # create synapses
    s_inp_exc = Connections(spike_gen, n_exc,
                            equation_builder=synapse_eq_builder(),
                            method="euler",
                            name=groupname + '_' + 's_inp_exc')
    s_exc_exc = Connections(n_exc, n_exc,
                            equation_builder=synapse_eq_builder(),
                            method="euler",
                            name=groupname + '_' + 's_exc_exc')
    s_inh_exc = Connections(n_inh, n_exc,
                            equation_builder=synapse_eq_builder(),
                            method="euler",
                            name=groupname + '_' + 's_inh_exc')
    s_exc_inh = Connections(n_exc, n_inh,
                            equation_builder=synapse_eq_builder(),
                            method="euler",
                            name=groupname + '_' + 's_exc_inh')
    s_inh_inh = Connections(n_inh, n_inh,
                            equation_builder=synapse_eq_builder(),
                            method='euler',
                            name=groupname + '_' + 's_inh_inh')

    # connect synapses
    s_inp_exc.connect('i==j')
    # connect the nearest neighbors including itself
    s_exc_exc.connect('abs(i-j)<=cutoff')
    # Generates all to all connectivity with specified probability of
    # connection
    s_exc_inh.connect('True', p=ei_connection_probability)
    s_inh_exc.connect('True', p=ie_connection_probability)
    s_inh_inh.connect('True', p=ii_connection_probability)

    s_exc_exc.add_state_variable(
        name='lateral_weight', shared=True, constant=True)
    s_exc_exc.add_state_variable(
        name='lateral_sigma', shared=True, constant=True)

    # set weights
    s_inp_exc.weight = we_inp_exc
    s_exc_inh.weight = we_exc_inh
    s_inh_exc.weight = wi_inh_exc
    s_inh_inh.weight = wi_inh_inh
    # lateral excitation kernel
    # we add an additional attribute to that synapse, which allows us to
    # change and retrieve that value more easily
    s_exc_exc.lateral_weight = we_exc_exc
    s_exc_exc.lateral_sigma = sigm
    s_exc_exc.namespace.update({spatial_kernel_name: spatial_kernel_func})
    s_exc_exc.weight = 'lateral_weight * ' + \
        spatial_kernel_name + '(i,j,lateral_sigma)'

    _groups = {
        'n_exc': n_exc,
        'n_inh': n_inh,
        'spike_gen': spike_gen,
        's_inp_exc': s_inp_exc,
        's_exc_exc': s_exc_exc,
        's_exc_inh': s_exc_inh,
        's_inh_exc': s_inh_exc,
        's_inh_inh': s_inh_inh, }

    # spikemons
    if monitor:
        monitors = {}
        spikemon_exc = SpikeMonitor(
            n_exc, name=groupname + '_spikemon_exc')
        spikemon_inh = SpikeMonitor(
            n_inh, name=groupname + '_spikemon_inh')
        spikemon_inp = SpikeMonitor(
            spike_gen, name=groupname + '_spikemon_inp')
        try:
            statemon_exc = StateMonitor(n_exc, ('Vm', 'Iin'),
                                        record=True,
                                        name=groupname + '_statemon_exc')
        except KeyError:
            statemon_exc = StateMonitor(n_exc, ('Imem', 'Iin'),
                                        record=True,
                                        name=groupname + '_statemon_exc')
        monitors = {
            'spikemon_exc': spikemon_exc,
            'spikemon_inh': spikemon_inh,
            'spikemon_inp': spikemon_inp,
            'statemon_exc': statemon_exc}
    else:
        monitors = {}
    # replacevars should be the 'real' names of the parameters, that can be
    # changed by the arguments of this function:
    # in this case: we_inp_exc, we_exc_inh, wi_inh_exc, we_exc_exc,rp_exc,
    # rp_inh,sigm
    standalone_params = {
        s_inp_exc.name + '_weight': we_inp_exc,
        s_exc_inh.name + '_weight': we_exc_inh,
        s_inh_exc.name + '_weight': wi_inh_exc,
        s_exc_exc.name + '_lateral_weight': we_exc_exc,
        s_exc_exc.name + '_lateral_sigma': sigm,
        n_exc.name + '_refP': rp_exc,
        n_inh.name + '_refP': rp_inh,
    }

    end = time.time()
    if verbose:
        print('creating WTA of ' + str(num_neurons) + ' neurons with name ' +
              groupname + ' took ' + str(end - start) + ' sec')
        print('The keys of the output dict are:')
        for key in _groups:
            print(key)

    return _groups, monitors, standalone_params


def gen2dWTA(groupname,
             neuron_eq_builder=DPI,
             synapse_eq_builder=DPISyn,
             we_inp_exc=1.5, we_exc_inh=1, wi_inh_inh=-1,
             wi_inh_exc=-1, we_exc_exc=2.0,
             sigm=2.5, rp_exc=3 * ms, rp_inh=1 * ms,
             num_neurons=64, num_inh_neurons=5,
             num_input_neurons=None, num_inputs=1, num_inh_inputs=2,
             cutoff=10, spatial_kernel="kernel_gauss_2d",
             ei_connection_probability=1.0, ie_connection_probability=1.0,
             ii_connection_probability=0.1,
             additional_statevars=[], monitor=True, verbose=False):
    '''Creates a 2D square WTA population of neurons, including the
    inhibitory interneuron population

    Args:
        groupname (str, required): Name of the WTA population.
        neuron_eq_builder (class, optional): neuron class as imported from
            models/neuron_models.
        synapse_eq_builder (class, optional): synapse class as imported from
            models/synapse_models.
        we_inp_exc (float, optional): Excitatory synaptic weight between
            input SpikeGenerator and WTA neurons.
        we_exc_inh (int, optional): Excitatory synaptic weight between WTA
            population and inhibitory interneuron.
        wi_inh_inh (int, optional): Self-inhibitory weight of the
            interneuron population.
        wi_inh_exc (TYPE, optional): Inhibitory synaptic weight between
            inhibitory interneuron and WTA population.
        we_exc_exc (float, optional): Self-excitatory synaptic weight (WTA).
        sigm (int, optional): Standard deviation in number of neurons for
            Gaussian connectivity kernel.
        rp_exc (float, optional): Refractory period of WTA neurons.
        rp_inh (float, optional): Refractory period of inhibitory neurons.
        num_neurons (int, optional): Size of WTA neuron population.
        num_inh_neurons (int, optional): Size of inhibitory interneuron
            population.
        num_input_neurons (int, optional): Size of input population.
            If None, equal to size of WTA population.
        num_inputs (int, optional): Number of input currents to WTA.
        num_inh_inputs (int, optional): Number of input currents to the
            inhibitory group.
        cutoff (int, optional): Radius of self-excitation.
        spatial_kernel (str, optional): Description
        ei_connection_probability (float, optional): WTA to interneuron
            connectivity probability.

        ie_connection_probability (float, optional): Interneuron to excitory
            neuron connectivity probability.
        ii_connection_probability (float, optional): Interneuron to
            interneuron neuron connectivity probability.
        additional_statevars (list, optional): List of additional state
            variables which are not standard.
        monitor (bool, optional): Flag to auto-generate spike and
            statemonitors.
        verbose (bool, optional): Flag to gain additional information.

    Returns:
        _groups (dictionary): Keys to all neuron and synapse groups.
        monitors (dictionary): Keys to all spike and state monitors.
        standalone_params (dictionary): Dictionary which holds all
            parameters to create a standalone network.
    '''

    if spatial_kernel is None:
        spatial_kernel = "kernel_gauss_2d"

    if type(spatial_kernel) == core.functions.Function:
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
    n_exc = Neurons(num2dNeurons,
                    equation_builder=neuron_eq_builder(
                        num_inputs=3+num_inputs),
                    refractory=rp_exc,
                    name=groupname + '_n_exc')
    n_inh = Neurons(num_inh_neurons,
                    equation_builder=neuron_eq_builder(
                        num_inputs=num_inh_inputs),
                    refractory=rp_inh,
                    name=groupname + '_n_inh')

    n_exc.namespace['num_neurons'] = num_neurons
    n_exc.namespace['ind2x'] = ind2x
    n_exc.namespace['ind2y'] = ind2y
    n_exc.x = "ind2x(i, num_neurons,num_neurons)"
    n_exc.y = "ind2y(i, num_neurons,num_neurons)"

    if num_input_neurons is None:
        num_input2d_neurons = num2dNeurons
    else:
        num_input2d_neurons = num_input_neurons**2
    # empty input for WTA group
    ts = np.asarray([]) * ms
    ind = np.asarray([])
    spike_gen = SpikeGeneratorGroup(
        num_input2d_neurons, indices=ind, times=ts,
        name=groupname + '_' + 'spike_gen')

    # create synapses
    s_inp_exc = Connections(spike_gen, n_exc,
                            equation_builder=synapse_eq_builder(),
                            method="euler",
                            name=groupname + '_s_inp_exc')
    s_exc_exc = Connections(n_exc, n_exc,
                            equation_builder=synapse_eq_builder(),
                            method="euler",
                            name=groupname + '_s_exc_exc')
    s_inh_exc = Connections(n_inh, n_exc,
                            equation_builder=synapse_eq_builder(),
                            method="euler",
                            name=groupname + '_s_inh_exc')
    s_exc_inh = Connections(n_exc, n_inh,
                            equation_builder=synapse_eq_builder(),
                            method="euler",
                            name=groupname + '_s_exc_inh')
    s_inh_inh = Connections(n_inh, n_inh,
                            equation_builder=synapse_eq_builder(),
                            method='euler',
                            name=groupname + '_s_inh_inh')

    # connect synapses
    s_inp_exc.connect('i==j')
    # connect the nearest neighbors including itself
    s_exc_exc.connect('dist1d2dint(i,j,num_neurons,num_neurons)<=cutoff')
    # Generates all to all connectivity
    s_exc_inh.connect('True', p=ei_connection_probability)
    s_inh_exc.connect('True', p=ie_connection_probability)
    s_inh_inh.connect('True', p=ii_connection_probability)

    s_exc_exc.add_state_variable(
        name='lateral_weight', shared=True, constant=True)
    s_exc_exc.add_state_variable(
        name='lateral_sigma', shared=True, constant=True)

    # set weights
    s_inp_exc.weight = we_inp_exc
    s_exc_inh.weight = we_exc_inh
    s_inh_exc.weight = wi_inh_exc
    s_inh_inh.weight = wi_inh_inh

    # lateral excitation kernel
    # we add an additional attribute to that synapse, which
    # allows us to change and retrieve that value more easily
    s_exc_exc.lateral_weight = we_exc_exc
    s_exc_exc.lateral_sigma = sigm
    s_exc_exc.namespace[spatial_kernel_name] = spatial_kernel_func
    s_exc_exc.namespace['num_neurons'] = num_neurons
    s_exc_exc.weight = 'lateral_weight * ' + spatial_kernel_name + \
        '(i,j,lateral_sigma,num_neurons,num_neurons)'

    _groups = {
        'n_exc': n_exc,
        'n_inh': n_inh,
        'spike_gen': spike_gen,
        's_inp_exc': s_inp_exc,
        's_exc_exc': s_exc_exc,
        's_exc_inh': s_exc_inh,
        's_inh_exc': s_inh_exc,
        's_inh_inh': s_inh_inh}

    if monitor:
        spikemon_exc = SpikeMonitor(
            n_exc, name=groupname + '_spikemon_exc')
        spikemon_inh = SpikeMonitor(
            n_inh, name=groupname + '_spikemon_inh')
        spikemon_inp = SpikeMonitor(
            spike_gen, name=groupname + '_spikemon_inp')
        try:
            statemon_exc = StateMonitor(n_exc, ('Vm', 'Iin'),
                                        record=True,
                                        name=groupname + '_statemon_exc')
        except KeyError:
            statemon_exc = StateMonitor(n_exc, ('Imem', 'Iin'),
                                        record=True,
                                        name=groupname + '_statemon_exc')
        monitors = {
            'spikemon_exc': spikemon_exc,
            'spikemon_inh': spikemon_inh,
            'spikemon_inp': spikemon_inp,
            'statemon_exc': statemon_exc}

    else:
        monitors = {}

    # replacevars should be the real names of the parameters,
    # that can be changed by the arguments of this function:
    # in this case: we_inp_exc, we_exc_inh, wi_inh_exc, we_exc_exc,rp_exc,
    # rp_inh,sigm
    standalone_params = {
        s_inp_exc.name + '_weight': we_inp_exc,
        s_exc_inh.name + '_weight': we_exc_inh,
        s_inh_exc.name + '_weight': wi_inh_exc,
        s_inh_inh.name + '_weight': wi_inh_inh,
        s_exc_exc.name + '_lateral_weight': we_exc_exc,
        s_exc_exc.name + '_lateral_sigma': sigm,
        n_exc.name + '_refP': rp_exc,
        n_inh.name + '_refP': rp_inh,
    }

    end = time.time()
    if verbose:
        print('creating WTA of ' + str(num_neurons) + ' x ' +
              str(num_neurons) + ' neurons with name ' +
              groupname + ' took ' + str(end - start) + ' sec')
        print('The keys of the ' + groupname + ' output dict are:')
        for key in _groups:
            print(key)

    return _groups, monitors, standalone_params


def set_wta_tags(self, _groups):
    """Sets default tags to a WTA network

    Args:
        _groups (dictionary): Keys to all neuron and synapse groups.

    No Longer Returned:
        Tags will be added to all _groups passed. They follow this structure:

        tags = {'mismatch' : (bool, 0/1)
                'noise : (bool, 0/1)
                'level': int
                'sign': str (exc/inh/None)
                'target sign': str (exc/inh/None)
                'num_inputs' : int (0 if not Neuron group),
                'bb_type' : str (WTA/3-WAY),
                'group_type' : str (Neuron/Connection/ SpikeGen)
                'connection_type' : str (rec/lateral/fb/ff/None)
            }
    """

    self._set_tags(tags_parameters.basic_wta_n_exc, _groups['n_exc'])
    self._set_tags(tags_parameters.basic_wta_n_inh, _groups['n_inh'])
    self._set_tags(tags_parameters.basic_wta_n_sg, _groups['spike_gen'])
    self._set_tags(tags_parameters.basic_wta_s_exc_exc, _groups['s_exc_exc'])
    self._set_tags(tags_parameters.basic_wta_s_exc_inh, _groups['s_exc_inh'])
    self._set_tags(tags_parameters.basic_wta_s_inh_exc, _groups['s_inh_exc'])
    self._set_tags(tags_parameters.basic_wta_s_inh_inh, _groups['s_inh_inh'])
    self._set_tags(tags_parameters.basic_wta_s_inp_exc, _groups['s_inp_exc'])
