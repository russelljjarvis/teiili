# -*- coding: utf-8 -*-
# @Author: mmilde, alpren
# @Date:   2017-12-27 10:46:44
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-18 17:15:23

"""
This files contains different Reservoir circuits
NicolaClopath2017
...
"""

import time
import numpy as np
# import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

from brian2 import ms, SpikeGeneratorGroup, SpikeMonitor,\
    StateMonitor, figure, subplot, mV, pA

from NCSBrian2Lib.tools.misc import print_states, dist1d2dint
from NCSBrian2Lib.tools.indexing import ind2x, ind2y
from NCSBrian2Lib.tools.plotting import plot_spikemon_qt, plot_statemon_qt

from NCSBrian2Lib.building_blocks.building_block import BuildingBlock
from NCSBrian2Lib.core.groups import Neurons, Connections

from NCSBrian2Lib.models.neuron_models import Izhikevich
from NCSBrian2Lib.models.synapse_models import DPISyn

from NCSBrian2Lib.models.parameters.izh_neuron_param import parameters as IzhParams

# RParams = {'weInpR': 1.5,
#            'weRInh': 1,
#            'wiInhR': -1,
#            'weRR': 0.5,
#            'sigm': 3,
#            'rpR': 3 * ms,
#            'rpInh': 1 * ms
# }


class R(BuildingBlock):
    '''A recurrent Neural Net inmplementing a Reservoir

    Attributes:
        group (dict): List of keys of neuron population
        inputGroup (SpikeGenerator): SpikeGenerator obj. to stimulate R
        num_neurons (int, optional): Size of R neuron population
        fraction_inh_neurons (float, optional): Set to None to skip Dale's priciple 
        spikemonR (TYPE): Description
        standaloneParams (dict): Keys for all standalone parameters necessary for cpp code generation
    '''

    def __init__(self, name,
                 neuron_eq_builder=Izhikevich,
                 synapse_eq_builder=ExpSyn,
                 block_params=wtaParams,
                 num_neurons=16,
                 fraction_inh_neurons=None,
                 additional_statevars=[],
                 num_inputs=1,
                 monitor=True,
                 debug=False):
        """Summary

        Args:
            groupname (str, required): Name of the R population
            dimensions (int, optional): Specifies if 1 or 2 dimensional R is created
            neuron_eq_builder (class, optional): neuron class as imported from models/neuron_models
            synapse_eq_builder (class, optional): synapse class as imported from models/synapse_models
            block_params (dict, optional): Parameter for neuron populations
            num_neurons (int, optional): Size of R neuron population
            fraction_inh_neurons (float, optional): Set to None to skip Dale's priciple
            additional_statevars (list, optional): List of additonal statevariables which are not standard
            num_inputs (int, optional): Number of input currents to R
            monitor (bool, optional): Flag to auto-generate spike and statemonitors
            debug (bool, optional): Flag to gain additional information

        Raises:
            NotImplementedError: 
        """
        self.num_neurons = num_neurons
        BuildingBlock.__init__(self, name,
                               neuron_eq_builder,
                               synapse_eq_builder,
                               block_params,
                               debug,
                               monitor)

        self.Groups, self.Monitors, self.standaloneParams = genR(name,
                                                                   neuron_eq_builder,
                                                                   synapse_eq_builder,
                                                                   num_neurons=num_neurons,
                                                                   fraction_inh_neurons=fraction_inh_neurons,
                                                                   additional_statevars=additional_statevars,
                                                                   cutoff=cutoff,
                                                                   num_inputs=num_inputs,
                                                                   monitor=monitor,
                                                                   debug=debug,
                                                                   **block_params)

        self.inputGroup =self.Groups['gRInpGroup']
        self.group = self.Groups['gRGroup']
        if monitor:
            self.spikemonR = self.Monitors['spikemonR']

    def plot(self, start_time=0 * ms, end_time=None):
        """Simple plot for R

        Args:
            start_time (int, optional): Start time of plot in ms
            end_time (int, optional): End time of plot in ms
        """

        if end_time is None:
            if len(self.spikemonR.t) > 0:
                end_time = max(self.spikemonR.t)
            else:
                end_time = end_time * ms
        plotR(self.name, start_time, end_time, self.numNeurons **
                self.dimensions, self.Monitors)


def genR(groupname,
             neuron_eq_builder=ExpAdaptIF,
             synapse_eq_builder=DPISyn,
             weInpR=1.5, weRInh=1, wiInhR=-1, weRR=0.5, sigm=3,
             rpR=3 * ms, rpInh=1 * ms,
             num_neurons=64, fraction_inh_neurons=0.2, num_inputs=1,
             monitor=True, additional_statevars=[], debug=False):
    """Summary

    Args:
        groupname (str, required): Name of the R population
        neuron_eq_builder (class, optional): neuron class as imported from models/neuron_models
        synapse_eq_builder (class, optional): synapse class as imported from models/synapse_models
        weInpR (float, optional): Excitatory synaptic weight between input SpikeGenerator and R neurons
        weRInh (int, optional): Excitatory synaptic weight between R population and inhibitory interneuron
        wiInhR (TYPE, optional): Inhibitory synaptic weight between inhibitory interneuron and R population
        weRR (float, optional): Self-excitatory synaptic weight (R)
        sigm (int, optional): Description
        rpR (float, optional): Refractory period of R neurons
        rpInh (float, optional): Refractory period of inhibitory neurons
        num_neurons (int, optional): Size of R neuron population
        fraction_inh_neurons (int, optional): Set to None to skip Dale's priciple
        cutoff (int, optional): Radius of self-excitation
        num_inputs (int, optional): Number of input currents to R
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
    gRGroup = Neurons(num_neurons, equation_builder=neuron_eq_builder(),
                        refractory=rpR, name='g' + groupname,
                        num_inputs=3 + num_inputs)
    if fraction_inh_neurons is not None:
        gRInhGroup = Neurons(round(num_neurons*fraction_inh_neurons), equation_builder=neuron_eq_builder(),
                             refractory=rpInh, name='g' + groupname + '_Inh',
                             num_inputs=1)

    # empty input for R group
    tsR = np.asarray([]) * ms
    indR = np.asarray([])
    gRInpGroup = SpikeGeneratorGroup(
        num_neurons, indices=indR, times=tsR, name='g' + groupname + '_Inp')

    # create synapses
    synInpR = Connections(gRInpGroup, gRGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_Inp')
    synRR = Connections(gRGroup, gRGroup,
                              equation_builder=synapse_eq_builder(),
                              method="euler", name='s' + groupname + '_RR')
    if fraction_inh_neurons is not None:
        synInhR1i = Connections(gRInhGroup, gRGroup,
                                equation_builder=synapse_eq_builder(),
                                method="euler", name='s' + groupname + '_Inhi')
        synRInh1e = Connections(gRGroup, gRInhGroup,
                                equation_builder=synapse_eq_builder(),
                                method="euler", name='s' + groupname + '_Inhe')

    # connect synapses
    synInpR.connect(p=0.1)
    # connect the nearest neighbors including itself
    synRR.connect(p=0.1)
    if fraction_inh_neurons is not None:
        synRInh1e.connect(p=0.1)  # Generates all to all connectivity
        synInhR1i.connect(p=0.1)

    synRR.addStateVariable(name='latWeight', shared=True, constant=True)
    synRR.addStateVariable(name='latSigma', shared=True, constant=True)

    # set weights
    synInpR.weight = weInpR
    synRInh1e.weight = weRInh
    synInhR1i.weight = wiInhR
    # lateral excitation kernel
    # we add an additional attribute to that synapse, which allows us to change
    # and retrieve that value more easily
    synRR1e.latWeight = weRR
    synRR1e.latSigma = sigm
    synRR1e.namespace['kernel_mexican_1d'] = kernel_mexican_1d
    synRR1e.weight = 'latWeight * kernel_mexican_1d(i,j,latSigma)'

    Groups = {
        'gRGroup': gRGroup,
        'gRInhGroup': gRInhGroup,
        'gRInpGroup': gRInpGroup,
        'synInpR': synInpR,
        'synRR': synRR}
    if fraction_inh_neurons is not None:
        Groups['synRInh1e'] = synRInh1e
        Groups['synInhR1i'] = synInhR1i

    # spikemons
    if monitor:
        spikemonR = SpikeMonitor(gRGroup, name='spikemon' + groupname + '_R')
        if fraction_inh_neurons is not None:
            spikemonRInh = SpikeMonitor(gRInhGroup, name='spikemon' + groupname + '_RInh')
        spikemonRInp = SpikeMonitor(gRInpGroup, name='spikemon' + groupname + '_RInp')
        try:
            statemonR = StateMonitor(gRGroup, ('Vm', 'Ie', 'Ii'), record=True,
                                       name='statemon' + groupname + '_R')
        except KeyError:
            statemonR = StateMonitor(gRGroup, ('Imem', 'Iin'), record=True,
                                       name='statemon' + groupname + '_R')
        Monitors = {
            'spikemonR': spikemonR,
            'spikemonRInh': spikemonRInh,
            'spikemonRInp': spikemonRInp,
            'statemonR': statemonR}

    # replacevars should be the 'real' names of the parameters, that can be
    # changed by the arguments of this function:
    # in this case: weInpR, weRInh, wiInhR, weRR,rpR, rpInh,sigm
    standaloneParams = {
        synInpR1e.name + '_weight': weInpR,
        synRInh1e.name + '_weight': weRInh,
        synInhR1i.name + '_weight': wiInhR,
        synRR1e.name + '_latWeight': weRR,
        synRR1e.name + '_latSigma': sigm,
        gRGroup.name + '_refP': rpR,
        gRInhGroup.name + '_refP': rpInh,
    }

    end = time.clock()
    if debug:
        print('creating R of ' + str(num_neurons) + ' neurons with name ' +
              groupname + ' took ' + str(end - start) + ' sec')
        print('The keys of the output dict are:')
        for key in Groups:
            print(key)

    return Groups, Monitors, standaloneParams


def plotR(name, start_time, end_time, num_neurons, RMonitors):
    """Function to easily visualize R activity.

    Args:
        name (str, required): Name of the R population
        start_time (brian2.units.fundamentalunits.Quantity, required): Start time in ms
            from when network activity should be plotted.
        end_time (brian2.units.fundamentalunits.Quantity, required): End time in ms of plot.
            Can be smaller than simulation time but not larger
        num_neurons (int, required): 1D number of neurons in R populations
        RMonitors (dict.): Dictionary with keys to access spike- and statemonitors. in R.Monitors
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
    raster_wta = win_raster.addPlot(title="SpikeMonitor R")
    win_raster.nextRow()
    raster_inh = win_raster.addPlot(title="SpikeMonitor inhibitory interneurons")

    state_membrane = win_states.addPlot(title='StateMonitor membrane potential')
    win_states.nextRow()
    state_syn_input = win_states.addPlot(title="StateMonitor synaptic input")

    plot_spikemon_qt(start_time=start_time, end_time=end_time,
                     num_neurons=16, monitor=RMonitors['spikemonRInp'], window=raster_input)
    plot_spikemon_qt(start_time=start_time, end_time=end_time,
                     num_neurons=16, monitor=RMonitors['spikemonR'], window=raster_wta)
    plot_spikemon_qt(start_time=start_time, end_time=end_time,
                     num_neurons=16, monitor=RMonitors['spikemonRInh'], window=raster_inh)

    plot_statemon_qt(start_time=start_time, end_time=end_time,
                     monitor=RMonitors['statemonR'], neuron_id=128,
                     variable="Imem", unit=pA, window=state_membrane, name=name)
    plot_statemon_qt(start_time=start_time, end_time=end_time,
                     monitor=RMonitors['statemonR'], neuron_id=128,
                     variable="Iin", unit=pA, window=state_syn_input, name=name)

    QtGui.QApplication.instance().exec_()

    # fig = figure(figsize=(8, 3))
    # plotSpikemon(start_time, end_time,
    #              RMonitors['spikemonR'], num_neurons, ylab='ind R_' + name)
    # fig = figure(figsize=(8, 3))
    # plotSpikemon(start_time, end_time,
    #              RMonitors['spikemonRInp'], None, ylab='ind RInp_' + name)
    # fig = figure(figsize=(8, 3))
    # plotSpikemon(start_time, end_time,
    #              RMonitors['spikemonRInh'], None, ylab='ind RInh_' + name)
    # # fig.savefig('fig/'+name+'_Spikes.png')

    # if num_neurons > 20:
    #     plot_state_neurons = range(20)
    # else:
    #     plot_state_neurons = range(num_neurons)

    # statemonR = RMonitors['statemonR']
    # if len(statemonR.t) > 0:
    #     fig = figure(figsize=(8, 10))
    #     nPlots = 3 * 100
    #     subplot(nPlots + 11)
    #     for ii in plot_state_neurons:
    #         plotStatemon(start_time, end_time, statemonR,
    #                      ii, variable='Imem', unit=pA, name=name)
    #     subplot(nPlots + 12)
    #     for ii in plot_state_neurons:
    #         plotStatemon(start_time, end_time, statemonR,
    #                      ii, variable='Iin', unit=pA, name=name)
    #     # subplot(nPlots + 13)
    #     # for ii in plot_state_neurons:
    #     #     plotStatemon(start_time, end_time, statemonR,
    #     #                  ii, variable='Ie1', unit=pA, name=name)
    #     # fig.savefig('fig/'+name+'_States.png', dpi=300)
    # plt.draw()
    # plt.show()
