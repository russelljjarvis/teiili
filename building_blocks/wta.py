# -*- coding: utf-8 -*-
# @Author: mmilde, alpren
# @Date:   2017-12-27 10:46:44
# @Last Modified by:   mmilde
# @Last Modified time: 2018-03-31 14:17:56

"""
This files contains different WTA circuits
1dWTA
2dWTA
"""

import time
import numpy as np
# import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

from brian2 import ms, SpikeGeneratorGroup, SpikeMonitor,\
    StateMonitor, figure, subplot, mV, pA

from NCSBrian2Lib.tools.synaptic_kernel import kernel_mexican_1d, kernel_mexican_2d
from NCSBrian2Lib.tools.misc import print_states, dist1d2dint
from NCSBrian2Lib.tools.indexing import ind2x, ind2y
from NCSBrian2Lib.tools.plotting import plot_spikemon_qt, plot_statemon_qt

from NCSBrian2Lib.building_blocks.building_block import BuildingBlock
from NCSBrian2Lib.core.groups import Neurons, Connections

from NCSBrian2Lib.models.neuron_models import DPI
from NCSBrian2Lib.models.synapse_models import DPISyn


wtaParams = {'weInpWTA': 1.5,
             'weWTAInh': 1,
             'wiInhWTA': -1,
             'weWTAWTA': 0.5,
             'sigm': 3,
             'rpWTA': 3 * ms,
             'rpInh': 1 * ms
             }


class WTA(BuildingBlock):
    '''a 1 or 2D square WTA

    Attributes:
        dimensions (int, optional): Specifies if 1 or 2 dimensional WTA is created
        group (dict): List of keys of neuron population
        inputGroup (SpikeGenerator): SpikeGenerator obj. to stimulate WTA
        num_neurons (int, optional): Size of WTA neuron population
        spikemonWTA (TYPE): Description
        standaloneParams (dict): Keys for all standalone parameters necessary for cpp code generation
    '''

    def __init__(self, name,
                 dimensions=1,
                 neuron_eq_builder=DPI,
                 synapse_eq_builder=DPISyn,
                 block_params=wtaParams,
                 num_neurons=16,
                 num_inh_neurons=2,
                 num_input_neurons=None,
                 cutoff=10,
                 additional_statevars=[],
                 num_inputs=1,
                 monitor=True,
                 debug=False):
        """Summary

        Args:
            groupname (str, required): Name of the WTA population
            dimensions (int, optional): Specifies if 1 or 2 dimensional WTA is created
            neuron_eq_builder (class, optional): neuron class as imported from models/neuron_models
            synapse_eq_builder (class, optional): synapse class as imported from models/synapse_models
            block_params (dict, optional): Parameter for neuron populations
            num_neurons (int, optional): Size of WTA neuron population
            num_inh_neurons (int, optional): Size of inhibitory interneuron population
            num_input_neurons (int, optional): Size of input population. If None size will be equal WTA population
            cutoff (int, optional): Radius of self-excitation
            additional_statevars (list, optional): List of additonal statevariables which are not standard
            num_inputs (int, optional): Number of input currents to WTA
            monitor (bool, optional): Flag to auto-generate spike and statemonitors
            debug (bool, optional): Flag to gain additional information

        Raises:
            NotImplementedError: If dimension is set larger than 2 error is raised
        """
        self.num_neurons = num_neurons
        self.dimensions = dimensions
        BuildingBlock.__init__(self, name,
                               neuron_eq_builder,
                               synapse_eq_builder,
                               block_params,
                               debug,
                               monitor)

        if dimensions == 1:
            self.Groups, self.Monitors, self.standaloneParams = gen1dWTA(name,
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
                                             **block_params)
        elif dimensions == 2:
            self.Groups, self.Monitors, self.standaloneParams = gen2dWTA(name,
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
                                             **block_params)

        else:
            raise NotImplementedError("only 1 and 2 d WTA available, sorry")

        self.inputGroup = self.Groups['gWTAInpGroup']
        self.group = self.Groups['gWTAGroup']
        if monitor:
            self.spikemonWTA = self.Monitors['spikemonWTA']

    def plot(self, start_time=0 * ms, end_time=None):
        """Simple plot for WTA

        Args:
            start_time (int, optional): Start time of plot in ms
            end_time (int, optional): End time of plot in ms
        """

        if end_time is None:
            if len(self.spikemonWTA.t) > 0:
                end_time = max(self.spikemonWTA.t)
            else:
                end_time = end_time * ms
        plotWTA(self.name, start_time, end_time, self.numNeurons **
                self.dimensions, self.Monitors)

# TODO: Generalize for n dimensions


def gen1dWTA(groupname,
             neuron_eq_builder=DPI,
             synapse_eq_builder=DPISyn,
             weInpWTA=1.5, weWTAInh=1, wiInhWTA=-1, weWTAWTA=0.5, sigm=3,
             rpWTA=3 * ms, rpInh=1 * ms,
             num_neurons=64, num_inh_neurons=5, num_input_neurons=None, cutoff=10, num_inputs=1,
             monitor=True, additional_statevars=[], debug=False):
    """Summary

    Args:
        groupname (str, required): Name of the WTA population
        neuron_eq_builder (class, optional): neuron class as imported from models/neuron_models
        synapse_eq_builder (class, optional): synapse class as imported from models/synapse_models
        weInpWTA (float, optional): Excitatory synaptic weight between input SpikeGenerator and WTA neurons
        weWTAInh (int, optional): Excitatory synaptic weight between WTA population and inhibitory interneuron
        wiInhWTA (TYPE, optional): Inhibitory synaptic weight between inhibitory interneuron and WTA population
        weWTAWTA (float, optional): Self-excitatory synaptic weight (WTA)
        sigm (int, optional): Description
        rpWTA (float, optional): Refractory period of WTA neurons
        rpInh (float, optional): Refractory period of inhibitory neurons
        num_neurons (int, optional): Size of WTA neuron population
        num_inh_neurons (int, optional): Size of inhibitory interneuron population
        num_input_neurons (int, optional): Size of input population. If None size will be equal WTA population
        cutoff (int, optional): Radius of self-excitation
        num_inputs (int, optional): Number of input currents to WTA
        monitor (bool, optional): Flag to auto-generate spike and statemonitors
        additional_statevars (list, optional): List of additonal statevariables which are not standard
        debug (bool, optional): Flag to gain additional information

    Returns:
        Groups (dictionary): Keys to all neuron and synapse groups
        Monitors (dictionary): Keys to all spike- and statemonitors
        standaloneParams (dictionary): Dictionary which holds all parameters to create a standalone network
    """
    # time measurement
    start = time.clock()

    # create neuron groups
    gWTAGroup = Neurons(num_neurons, equation_builder=neuron_eq_builder(),
                        refractory=rpWTA, name='g' + groupname,
                        num_inputs=3 + num_inputs)
    gWTAInhGroup = Neurons(num_inh_neurons, equation_builder=neuron_eq_builder(),
                           refractory=rpInh, name='g' + groupname + '_Inh',
                           num_inputs=1)

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
                              method="euler", name='s' + groupname + '_e')  # ,
    synInhWTA1i = Connections(gWTAInhGroup, gWTAGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inhi')
    synWTAInh1e = Connections(gWTAGroup, gWTAInhGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inhe')

    # connect synapses
    synInpWTA1e.connect('i==j')
    # connect the nearest neighbors including itself
    synWTAWTA1e.connect('abs(i-j)<=cutoff')
    synWTAInh1e.connect('True')  # Generates all to all connectivity
    synInhWTA1i.connect('True')

    synWTAWTA1e.addStateVariable(name='latWeight', shared=True, constant=True)
    synWTAWTA1e.addStateVariable(name='latSigma', shared=True, constant=True)

    # set weights
    synInpWTA1e.weight = weInpWTA
    synWTAInh1e.weight = weWTAInh
    synInhWTA1i.weight = wiInhWTA
    # lateral excitation kernel
    # we add an additional attribute to that synapse, which allows us to change
    # and retrieve that value more easily
    synWTAWTA1e.latWeight = weWTAWTA
    synWTAWTA1e.latSigma = sigm
    synWTAWTA1e.namespace['kernel_mexican_1d'] = kernel_mexican_1d
    synWTAWTA1e.weight = 'latWeight * kernel_mexican_1d(i,j,latSigma)'

    Groups = {
        'gWTAGroup': gWTAGroup,
        'gWTAInhGroup': gWTAInhGroup,
        'gWTAInpGroup': gWTAInpGroup,
        'synInpWTA1e': synInpWTA1e,
        'synWTAWTA1e': synWTAWTA1e,
        'synWTAInh1e': synWTAInh1e,
        'synInhWTA1i': synInhWTA1i}

    # spikemons
    if monitor:
        spikemonWTA = SpikeMonitor(gWTAGroup, name='spikemon' + groupname + '_WTA')
        spikemonWTAInh = SpikeMonitor(gWTAInhGroup, name='spikemon' + groupname + '_WTAInh')
        spikemonWTAInp = SpikeMonitor(gWTAInpGroup, name='spikemon' + groupname + '_WTAInp')
        try:
            statemonWTA = StateMonitor(gWTAGroup, ('Vm', 'Ie', 'Ii'), record=True,
                                       name='statemon' + groupname + '_WTA')
        except KeyError:
            statemonWTA = StateMonitor(gWTAGroup, ('Imem', 'Iin'), record=True,
                                       name='statemon' + groupname + '_WTA')
        Monitors = {
            'spikemonWTA': spikemonWTA,
            'spikemonWTAInh': spikemonWTAInh,
            'spikemonWTAInp': spikemonWTAInp,
            'statemonWTA': statemonWTA}

    # replacevars should be the 'real' names of the parameters, that can be
    # changed by the arguments of this function:
    # in this case: weInpWTA, weWTAInh, wiInhWTA, weWTAWTA,rpWTA, rpInh,sigm
    standaloneParams = {
        synInpWTA1e.name + '_weight': weInpWTA,
        synWTAInh1e.name + '_weight': weWTAInh,
        synInhWTA1i.name + '_weight': wiInhWTA,
        synWTAWTA1e.name + '_latWeight': weWTAWTA,
        synWTAWTA1e.name + '_latSigma': sigm,
        gWTAGroup.name + '_refP': rpWTA,
        gWTAInhGroup.name + '_refP': rpInh,
    }

    end = time.clock()
    if debug:
        print('creating WTA of ' + str(num_neurons) + ' neurons with name ' +
              groupname + ' took ' + str(end - start) + ' sec')
        print('The keys of the output dict are:')
        for key in Groups:
            print(key)

    return Groups, Monitors, standaloneParams


def gen2dWTA(groupname,
             neuron_eq_builder=DPI,
             synapse_eq_builder=DPISyn,
             weInpWTA=1.5, weWTAInh=1, wiInhWTA=-1, weWTAWTA=2, sigm=2.5,
             rpWTA=2.5 * ms, rpInh=1 * ms,
             wiInhInh=0, EI_connection_probability=1, IE_connection_probability=1,
             II_connection_probability=0.1,
             num_neurons=20, num_inh_neurons=3, num_input_neurons=None, cutoff=9, num_inputs=1,
             monitor=True, additional_statevars=[], debug=False):
    '''generates a new square 2d WTA

    Args:
        groupname (str, required): Name of the WTA population
        neuron_eq_builder (class, optional): neuron class as imported from models/neuron_models
        synapse_eq_builder (class, optional): synapse class as imported from models/synapse_models
        weInpWTA (float, optional): Excitatory synaptic weight between input SpikeGenerator and WTA neurons
        weWTAInh (int, optional): Excitatory synaptic weight between WTA population and inhibitory interneuron
        wiInhWTA (TYPE, optional): Inhibitory synaptic weight between inhibitory interneuron and WTA population
        weWTAWTA (float, optional): Self-excitatory synaptic weight (WTA)
        sigm (int, optional): Description
        rpWTA (float, optional): Refractory period of WTA neurons
        rpInh (float, optional): Refractory period of inhibitory neurons
        num_neurons (int, optional): Size of WTA neuron population
        num_inh_neurons (int, optional): Size of inhibitory interneuron population
        num_input_neurons (int, optional): Size of input population. If None size will be equal WTA population
        cutoff (int, optional): Radius of self-excitation
        num_inputs (int, optional): Number of input currents to WTA
        monitor (bool, optional): Flag to auto-generate spike and statemonitors
        additional_statevars (list, optional): List of additonal statevariables which are not standard
        debug (bool, optional): Flag to gain additional information

    Returns:
        Groups (dictionary): Keys to all neuron and synapse groups
        Monitors (dictionary): Keys to all spike- and statemonitors
        standaloneParams (dictionary): Dictionary which holds all parameters to create a standalone network
    '''

    # time measurement
    start = time.clock()

    # create neuron groups
    num2dNeurons = num_neurons**2
    num_inh_inputs = 2
    gWTAGroup = Neurons(num2dNeurons, equation_builder=neuron_eq_builder(),
                        refractory=rpWTA, name='g' + groupname,
                        num_inputs=3 + num_inputs)
    gWTAInhGroup = Neurons(num_inh_neurons, equation_builder=neuron_eq_builder(),
                           refractory=rpInh, name='g' + groupname + '_Inh',
                           num_inputs=num_inh_inputs)

    gWTAGroup.namespace['num_neurons'] = num_neurons
    gWTAGroup.namespace['ind2x'] = ind2x
    gWTAGroup.namespace['ind2y'] = ind2y
    gWTAGroup.x = "ind2x(i, num_neurons)"
    gWTAGroup.y = "ind2y(i, num_neurons)"

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
                              method="euler", name='s' + groupname + '_e')  # ,
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
    synWTAWTA1e.connect('dist1d2dint(i,j,num_neurons)<=cutoff')
    synWTAInh1e.connect('True', p=EI_connection_probability)  # Generates all to all connectivity
    synInhWTA1i.connect('True', p=IE_connection_probability)
    synInhInh1i.connect('True', p=II_connection_probability)

    synWTAWTA1e.addStateVariable(name='latWeight', shared=True, constant=True)
    synWTAWTA1e.addStateVariable(name='latSigma', shared=True, constant=True)

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
    synWTAWTA1e.namespace['kernel_mexican_2d'] = kernel_mexican_2d
    synWTAWTA1e.namespace['num_neurons'] = num_neurons
    synWTAWTA1e.weight = 'latWeight * kernel_mexican_2d(i,j,latSigma,num_neurons)'

    Groups = {
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
    spikemonWTAInh = SpikeMonitor(gWTAInhGroup, name='spikemon' + groupname + '_WTAInh')
    spikemonWTAInp = SpikeMonitor(gWTAInpGroup, name='spikemon' + groupname + '_WTAInp')
    try:
        statemonWTA = StateMonitor(gWTAGroup, ('Vm', 'Ie', 'Ii'), record=True,
                                   name='statemon' + groupname + '_WTA')
    except KeyError:
        statemonWTA = StateMonitor(gWTAGroup, ('Imem', 'Iin'), record=True,
                                   name='statemon' + groupname + '_WTA')
    Monitors = {
        'spikemonWTA': spikemonWTA,
        'spikemonWTAInh': spikemonWTAInh,
        'spikemonWTAInp': spikemonWTAInp,
        'statemonWTA': statemonWTA}

    # replacevars should be the real names of the parameters,
    # that can be changed by the arguments of this function:
    # in this case: weInpWTA, weWTAInh, wiInhWTA, weWTAWTA,rpWTA, rpInh,sigm
    standaloneParams = {
        synInpWTA1e.name + '_weight': weInpWTA,
        synWTAInh1e.name + '_weight': weWTAInh,
        synInhWTA1i.name + '_weight': wiInhWTA,
        synInhInh1i.name + '_weight': wiInhInh,
        synWTAWTA1e.name + '_latWeight': weWTAWTA,
        synWTAWTA1e.name + '_latSigma': sigm,
        gWTAGroup.name + '_refP': rpWTA,
        gWTAInhGroup.name + '_refP': rpInh,
    }

    end = time.clock()
    if debug:
        print ('creating WTA of ' + str(num_neurons) + ' x ' + str(num_neurons) + ' neurons with name ' +
               groupname + ' took ' + str(end - start) + ' sec')
        print('The keys of the output dict are:')
        for key in Groups:
            print(key)

    return Groups, Monitors, standaloneParams


def plotWTA(name, start_time, end_time, WTAMonitors):
    """Function to easily visualize WTA activity.

    Args:
        name (str, required): Name of the WTA population
        start_time (brian2.units.fundamentalunits.Quantity, required): Start time in ms
            from when network activity should be plotted.
        end_time (brian2.units.fundamentalunits.Quantity, required): End time in ms of plot.
            Can be smaller than simulation time but not larger
        WTAMonitors (dict.): Dictionary with keys to access spike- and statemonitors. in WTA.Monitors
    """
    pg.setConfigOptions(antialias=True)

    win_raster = pg.GraphicsWindow(title='Winner-Take-All Test Simulation: Raster plots')
    win_states = pg.GraphicsWindow(title='Winner-Take-All Test Simulation:State plots')
    win_raster.resize(1000, 1800)
    win_states.resize(1000, 1800)
    win_raster.setWindowTitle('Winner-Take-All Test Simulation: Raster plots')
    win_states.setWindowTitle('Winner-Take-All Test Simulation:State plots')

    raster_input = win_raster.addPlot(title="SpikeGenerator input")
    win_raster.nextRow()
    raster_wta = win_raster.addPlot(title="SpikeMonitor WTA")
    win_raster.nextRow()
    raster_inh = win_raster.addPlot(title="SpikeMonitor inhibitory interneurons")

    state_membrane = win_states.addPlot(title='StateMonitor membrane potential')
    win_states.nextRow()
    state_syn_input = win_states.addPlot(title="StateMonitor synaptic input")

    plot_spikemon_qt(start_time=start_time, end_time=end_time,
                     num_neurons=np.int(WTAMonitors['spikemonWTAInp'].source.N), monitor=WTAMonitors['spikemonWTAInp'],
                     window=raster_input)
    plot_spikemon_qt(start_time=start_time, end_time=end_time,
                     num_neurons=WTAMonitors['spikemonWTA'].source.N, monitor=WTAMonitors['spikemonWTA'],
                     window=raster_wta)
    plot_spikemon_qt(start_time=start_time, end_time=end_time,
                     num_neurons=WTAMonitors['spikemonWTAInh'].source.N, monitor=WTAMonitors['spikemonWTAInh'],
                     window=raster_inh)

    plot_statemon_qt(start_time=start_time, end_time=end_time,
                     monitor=WTAMonitors['statemonWTA'], neuron_id=128,
                     variable="Imem", unit=pA, window=state_membrane, name=name)
    plot_statemon_qt(start_time=start_time, end_time=end_time,
                     monitor=WTAMonitors['statemonWTA'], neuron_id=128,
                     variable="Iin", unit=pA, window=state_syn_input, name=name)

    QtGui.QApplication.instance().exec_()

    # fig = figure(figsize=(8, 3))
    # plotSpikemon(start_time, end_time,
    #              WTAMonitors['spikemonWTA'], num_neurons, ylab='ind WTA_' + name)
    # fig = figure(figsize=(8, 3))
    # plotSpikemon(start_time, end_time,
    #              WTAMonitors['spikemonWTAInp'], None, ylab='ind WTAInp_' + name)
    # fig = figure(figsize=(8, 3))
    # plotSpikemon(start_time, end_time,
    #              WTAMonitors['spikemonWTAInh'], None, ylab='ind WTAInh_' + name)
    # # fig.savefig('fig/'+name+'_Spikes.png')

    # if num_neurons > 20:
    #     plot_state_neurons = range(20)
    # else:
    #     plot_state_neurons = range(num_neurons)

    # statemonWTA = WTAMonitors['statemonWTA']
    # if len(statemonWTA.t) > 0:
    #     fig = figure(figsize=(8, 10))
    #     nPlots = 3 * 100
    #     subplot(nPlots + 11)
    #     for ii in plot_state_neurons:
    #         plotStatemon(start_time, end_time, statemonWTA,
    #                      ii, variable='Imem', unit=pA, name=name)
    #     subplot(nPlots + 12)
    #     for ii in plot_state_neurons:
    #         plotStatemon(start_time, end_time, statemonWTA,
    #                      ii, variable='Iin', unit=pA, name=name)
    #     # subplot(nPlots + 13)
    #     # for ii in plot_state_neurons:
    #     #     plotStatemon(start_time, end_time, statemonWTA,
    #     #                  ii, variable='Ie1', unit=pA, name=name)
    #     # fig.savefig('fig/'+name+'_States.png', dpi=300)
    # plt.draw()
    # plt.show()
