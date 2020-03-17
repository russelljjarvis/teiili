#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:09:49 2018

@author: dzenn
"""

import numpy as np

from brian2 import ms, SpikeMonitor,\
    prefs, PoissonGroup, Hz

from teili.building_blocks.building_block import BuildingBlock
from teili.building_blocks.wta import WTA
from teili.core.groups import Connections
from teili.core import tags as tags_parameters

from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn

from teili.tools.three_way_kernels import A_plus_B_equals_C
from teili.tools.visualizer.DataControllers import Rasterplot

try:
    # from pyqtgraph import QtGui
    from PyQt5 import QtGui
    import pyqtgraph as pg
    QtApp = QtGui.QApplication([])
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = False
except BaseException:
    SKIP_PYQTGRAPH_RELATED_UNITTESTS = True

threeway_params = {}

wta_params = {'we_inp_exc': 2000,
             'we_exc_inh': 300,
             'wi_inh_exc': -300,
             'we_exc_exc': 300,
             'sigm': 2.2,
             'rp_exc': 3 * ms,
             'rp_inh': 2 * ms,
             'ei_connection_probability': 1
             }


threeway_params.update(wta_params)


class Threeway(BuildingBlock):
    """A network of three 1d WTA populations connected to a hidden 2d WTA
    population implementing a three-way relation between three 1d quantities
    A, B and C (e.g. A + B = C) via a hidden population H.

    Attributes:
        Groups (dict): Complete list of keys of neuron groups, synapse groups
                       and WTA substructures of the Threeway building block to
                       be included into Network object for simulation
        monitors (dict): Complete list of Brian2 monitors for all entities of
                       the Threeway building block to be included into Network
                       object for simulation
        num_input_neurons (int): Sizes of input/output populations A, B and C
        num_neurons (int): Total amount of neurons used for the Threeway
                           structure including the hidden population H and
                           populations of inhibitory neurons used for WTA
                           connectivity
        A (WTA): A shortcut for a input/output population A implemented with
                 a Teili 1d WTA building block
        B (WTA): A shortcut for a input/output population B implemented with
                 a Teili 1d WTA building block
        C (WTA): A shortcut for a input/output population C implemented with
                 a Teili 1d WTA building block
        H (WTA): A shortcut for a hidden population H implemented with
                 a Teili 2d WTA building block
        Inp_A (PoissonGroup): PoissonGroup obj. to stimulate population A
        Inp_B (PoissonGroup): PoissonGroup obj. to stimulate population B
        Inp_C (PoissonGroup): PoissonGroup obj. to stimulate population C
        value_a (double): Stored input for A (center of a gaussian bump)
        value_b (double): Stored input for B (center of a gaussian bump)
        value_c (double): Stored input for C (center of a gaussian bump)
        standalone_params (dict): Keys for all standalone parameters necessary
                                  for cpp code generation (TBD)
    """

    def __init__(self, name,
                 neuron_eq_builder=DPI,
                 synapse_eq_builder=DPISyn,
                 block_params=threeway_params,
                 num_input_neurons=16,
                 num_hidden_neurons=256,
                 hidden_layer_gen_func=A_plus_B_equals_C,
                 cutoff=5,
                 additional_statevars=[],
                 spatial_kernel=None,
                 monitor=True,
                 debug=False):
        """
        Args:
            name (str, required): Name of the TW block
            neuron_eq_builder (class, optional): neuron class as imported
                                                 from models/neuron_models
            synapse_eq_builder (class, optional): synapse class as imported
                                                  from models/synapse_models
            block_params (dict, optional): Parameters for neuron populations
            num_input_neurons (int, optional): Sizes of input/output populations A, B and C
            num_hidden_neurons (int, optional): Size of the hidden population H
            hidden_layer_gen_func (class, optional): A class providing connectivity pattern
            cutoff (int, optional): connectivity kernel cutoff of WTAs
            additional_statevars (list, optional): List of additonal
                                                   statevariables which are
                                                   not standard
            monitor (bool, optional): Flag to auto-generate spike and
                                      statemonitors
            debug (bool, optional): Flag to gain additional information
        """
        self.num_input_neurons = num_input_neurons
        self.num_neurons = 3 * int(1.2 * num_input_neurons) + \
            int(1.2 * num_hidden_neurons)
        BuildingBlock.__init__(self, name,
                               neuron_eq_builder,
                               synapse_eq_builder,
                               block_params,
                               debug,
                               monitor)

        self.sub_blocks, self._groups, self.monitors, self.standalone_params = gen_threeway(name,
                                                                          neuron_eq_builder,
                                                                          synapse_eq_builder,
                                                                          num_input_neurons=num_input_neurons,
                                                                          num_hidden_neurons=num_hidden_neurons,
                                                                          hidden_layer_gen_func=hidden_layer_gen_func,
                                                                          additional_statevars=additional_statevars,
                                                                          cutoff=cutoff,
                                                                          spatial_kernel=spatial_kernel,
                                                                          monitor=monitor,
                                                                          debug=debug,
                                                                          block_params=block_params)
        
        set_TW_tags(self, self._groups)
        
        # Creating handles for neuron groups and inputs
        self.A = self.sub_blocks['wta_A']
        self.B = self.sub_blocks['wta_B']
        self.C = self.sub_blocks['wta_C']
        self.H = self.sub_blocks['wta_H']
        
        self.Inp_A = self.A._groups['spike_gen']
        self.Inp_B = self.B._groups['spike_gen']
        self.Inp_C = self.C._groups['spike_gen']

        self.value_a = np.NAN
        self.value_b = np.NAN
        self.value_c = np.NAN
        
        self.start_time = 0*ms
        
        self.input_groups.update({'A': self.A._groups['n_exc'],
                                  'B': self.B._groups['n_exc'],
                                  'C': self.C._groups['n_exc']})
        self.output_groups.update({'A': self.A._groups['n_exc'],
                                  'B': self.B._groups['n_exc'],
                                  'C': self.C._groups['n_exc']})
        self.hidden_groups.update({'H': self.H._groups['n_exc']})


    def set_A(self, value):
        """
        Sets spiking rates of neurons of the PoissonGroup Inp_A to satisfy
        a shape of a gaussian bump centered at 'value' between 0 and 1

        Args:
            value (float): a value to be encoded with an activity bump
        """

        self.Inp_A.rates = double2pop_code(value, self.num_input_neurons)
        self.value_a = value

    def set_B(self, value):
        """
        Sets spiking rates of neurons of the PoissonGroup Inp_B to satisfy
        a shape of a gaussian bump centered at 'value' between 0 and 1

        Args:
            value (float): a value to be encoded with an activity bump
        """

        self.Inp_B.rates = double2pop_code(value, self.num_input_neurons)
        self.value_b = value

    def set_C(self, value):
        """
        Sets spiking rates of neurons of the PoissonGroup Inp_C to satisfy
        a shape of a gaussian bump centered at 'value' between 0 and 1

        Args:
            value (float): a value to be encoded with an activity bump
        """

        self.Inp_C.rates = double2pop_code(value, self.num_input_neurons)
        self.value_c = value

    def reset_A(self):
        """
        Resets spiking rates of neurons of the PoissonGroup Inp_A to zero
        (e.g. turns the input A off)
        """
        self.Inp_A.rates = np.zeros(self.num_input_neurons) * Hz
        self.value_a = np.NAN

    def reset_B(self):
        """
        Resets spiking rates of neurons of the PoissonGroup Inp_B to zero
        (e.g. turns the input B off)
        """
        self.Inp_B.rates = np.zeros(self.num_input_neurons) * Hz
        self.value_b = np.NAN

    def reset_C(self):
        """
        Resets spiking rates of neurons of the PoissonGroup Inp_C to zero
        (e.g. turns the input C off)
        """
        self.Inp_C.rates = np.zeros(self.num_input_neurons) * Hz
        self.value_c = np.NAN

    def reset_inputs(self):
        """
        Resets all external inputs of the Threeway block
        """
        self.reset_A()
        self.reset_B()
        self.reset_C()

    def get_values(self, measurement_period=100 * ms):
        """
            Extracts encoded values of A, B and C from the spiking rates of
            the corresponding populations
    
            Args:
                measurement_period (ms, optional): Sets the interval back from
                current moment in time for the spikes to be included into
                rate calculation
        """

        if self.A.monitor is True and self.B.monitor is True and self.C.monitor is True:
            a = pop_code2double(get_rates(self.monitors['spikemon_A'],
                                          measurement_period=measurement_period))
            b = pop_code2double(get_rates(self.monitors['spikemon_B'],
                                          measurement_period=measurement_period))
            c = pop_code2double(get_rates(self.monitors['spikemon_C'],
                                          measurement_period=measurement_period))
            return a, b, c
        else:
            raise ValueError(
                'Unable to compute population vectors, monitoring has been\
                    turned off!')
            
    def plot(self):
        """
            Create a rasterplot of spikes of excitatory neurons of populations
            A, B and C
        """
        return plot_threeway_raster(self)
        
        


def gen_threeway(name,
                 neuron_eq_builder,
                 synapse_eq_builder,
                 block_params,
                 num_input_neurons,
                 num_hidden_neurons,
                 hidden_layer_gen_func,
                 additional_statevars,
                 cutoff,
                 spatial_kernel,
                 monitor,
                 debug):
    """
        Generator function for a Threeway building block
    """

    # TODO: Replace PoissonGroups as inputs with stimulus generators

    if debug:
        print("Creating WTA's!")

    wta_A = WTA(name + '_wta_A',
                dimensions=1,
                num_inputs=3,
                block_params=block_params,
                num_neurons=num_input_neurons,
                num_inh_neurons=int(0.2*num_input_neurons),
                cutoff=cutoff,
                monitor=True,
                verbose=debug)
    wta_B = WTA(name + '_wta_B',
                dimensions=1,
                num_inputs=3,
                block_params=block_params,
                num_neurons=num_input_neurons,
                num_inh_neurons=int(0.2 * num_input_neurons),
                cutoff=cutoff,
                monitor=True,
                verbose=debug)
    wta_C = WTA(name + '_wta_C',
                dimensions=1,
                num_inputs=3,
                block_params=block_params,
                num_neurons=num_input_neurons,
                num_inh_neurons=int(0.2 * num_input_neurons),
                cutoff=cutoff,
                monitor=True,
                verbose=debug)
    wta_H = WTA(name + '_wta_H',
                dimensions=2,
                num_inputs=3,
                block_params=block_params,
                num_neurons=num_input_neurons,
                num_inh_neurons=int(0.2 * num_hidden_neurons),
                cutoff=cutoff,
                monitor=monitor,
                verbose=debug)
    
    sub_blocks = {
        'wta_A': wta_A,
        'wta_B': wta_B,
        'wta_C': wta_C,
        'wta_H': wta_H}

    wta_A._groups['spike_gen'] = PoissonGroup(name=name + '_input_group_A',
        N=num_input_neurons, rates=np.zeros(num_input_neurons) * Hz)
    wta_B._groups['spike_gen'] = PoissonGroup(name=name + '_input_group_B',
        N=num_input_neurons, rates=np.zeros(num_input_neurons) * Hz)
    wta_C._groups['spike_gen'] = PoissonGroup(name=name + '_input_group_C',
        N=num_input_neurons, rates=np.zeros(num_input_neurons) * Hz)


    # Creating interpopulation synapse groups
    syn_A_H = Connections(wta_A.groups['n_exc'], wta_H.groups['n_exc'],
                          equation_builder=synapse_eq_builder(),
                          method="euler", name=name + '_s_A_to_H')
    syn_H_A = Connections(wta_H.groups['n_exc'], wta_A.groups['n_exc'],
                          equation_builder=synapse_eq_builder(),
                          method="euler", name=name + '_s_H_to_A')
    syn_B_H = Connections(wta_B.groups['n_exc'], wta_H.groups['n_exc'],
                          equation_builder=synapse_eq_builder(),
                          method="euler", name=name + '_s_B_to_H')
    syn_H_B = Connections(wta_H.groups['n_exc'], wta_B.groups['n_exc'],
                          equation_builder=synapse_eq_builder(),
                          method="euler", name=name + '_s_H_to_B')
    syn_C_H = Connections(wta_C.groups['n_exc'], wta_H.groups['n_exc'],
                          equation_builder=synapse_eq_builder(),
                          method="euler", name=name + '_s_C_to_H')
    syn_H_C = Connections(wta_H.groups['n_exc'], wta_C.groups['n_exc'],
                          equation_builder=synapse_eq_builder(),
                          method="euler", name= name + '_s_H_to_C')

    # Creating input synapse groups
    wta_A._groups['s_inp_exc'] = Connections(wta_A._groups['spike_gen'],
                 wta_A.groups['n_exc'], equation_builder=synapse_eq_builder(),
                            method="euler", name=name + '_s_inp_A')
    wta_B._groups['s_inp_exc'] = Connections(wta_B._groups['spike_gen'],
                 wta_B.groups['n_exc'], equation_builder=synapse_eq_builder(),
                            method="euler", name=name + '_s_inp_B')
    wta_C._groups['s_inp_exc'] = Connections(wta_C._groups['spike_gen'],
                 wta_C.groups['n_exc'], equation_builder=synapse_eq_builder(),
                            method="euler", name=name + '_s_inp_C')

    interPopSynGroups = {
        's_A_to_H' : syn_A_H,
        's_H_to_A' : syn_H_A,
        's_B_to_H' : syn_B_H,
        's_H_to_B' : syn_H_B,
        's_C_to_H' : syn_C_H,
        's_H_to_C' : syn_H_C}

    synGroups = {
            's_inp_A' : wta_A._groups['s_inp_exc'],
            's_inp_B' : wta_B._groups['s_inp_exc'],
            's_inp_C' : wta_C._groups['s_inp_exc']            
            }

    for tmp_syn_group in synGroups:
        synGroups[tmp_syn_group].connect('i == j')
        synGroups[tmp_syn_group].weight = wta_A.params['we_inp_exc']

    synGroups.update(interPopSynGroups)

    # Connecting the populations with a given index generation function
    # TODO: add more index generating functions
    index_gen_function = hidden_layer_gen_func

    for tmp_syn_group_name in interPopSynGroups:
        arr_i, arr_j = index_gen_function(
            tmp_syn_group_name[-6], tmp_syn_group_name[-1], num_input_neurons)
        interPopSynGroups[tmp_syn_group_name].connect(i=arr_i, j=arr_j)
        interPopSynGroups[tmp_syn_group_name].weight = wta_A.params['we_inp_exc']

    groups = {}
    groups.update(interPopSynGroups)

    if monitor:
        wta_A.monitors['spikemon_inp'] = SpikeMonitor(
            wta_A._groups['spike_gen'], name='spikemon' + name + '_InpA')
        wta_B.monitors['spikemon_inp'] = SpikeMonitor(
            wta_B._groups['spike_gen'], name='spikemon' + name + '_InpB')
        wta_C.monitors['spikemon_inp'] = SpikeMonitor(
            wta_C._groups['spike_gen'], name='spikemon' + name + '_InpC')

    monitors = {
        'spikemon_InpA': wta_A.monitors['spikemon_inp'],
        'spikemon_InpB': wta_B.monitors['spikemon_inp'],
        'spikemon_InpC': wta_C.monitors['spikemon_inp'],
        'spikemon_A': wta_A.monitors['spikemon_exc'],
        'spikemon_B': wta_B.monitors['spikemon_exc'],
        'spikemon_C': wta_C.monitors['spikemon_exc'],
        'statemon_A': wta_A.monitors['statemon_exc'],
        'statemon_B': wta_B.monitors['statemon_exc'],
        'statemon_C': wta_C.monitors['statemon_exc']}

    # monitors.update(wta_A.monitors, wta_B.monitors, wta_C.monitors, wta_H.monitors)

    standalone_params = {}

    return sub_blocks, groups, monitors, standalone_params

def set_TW_tags(TW_block, _groups):
    '''
    Sets default tags to members of the _groups of the Threeway block

    Args:
        _groups (dictionary): All neuron and synapse groups.

    Returns:
        _groups (dictionary): All neuron and synapse groups with tags appended.
    '''

    TW_block._set_tags(tags_parameters.basic_threeway_1WTA_to_2WTA,
                       _groups['s_A_to_H'])
    TW_block._set_tags(tags_parameters.basic_threeway_1WTA_to_2WTA,
                       _groups['s_B_to_H'])
    TW_block._set_tags(tags_parameters.basic_threeway_1WTA_to_2WTA,
                       _groups['s_C_to_H'])
    TW_block._set_tags(tags_parameters.basic_threeway_2WTA_to_1WTA,
                       _groups['s_H_to_A'])
    TW_block._set_tags(tags_parameters.basic_threeway_2WTA_to_1WTA,
                       _groups['s_H_to_B'])
    TW_block._set_tags(tags_parameters.basic_threeway_2WTA_to_1WTA,
                       _groups['s_H_to_C'])


def plot_threeway_raster(TW):
    """Function to easily visualize Threeway block activity.

    Args:
        TW (Threeway BuildingBlock, required): Threeway block to be visualized
        
    Returns:
        mainfig : pyqtgraph MainWindow of the plot
        plot_A, plot_B, plot_C : Rasterplot objects of teili visualizer
    """
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication()
    else:
        print('QApplication instance already exists: %s' % str(app))

    pg.setConfigOptions(antialias=True)

    mainfig = pg.GraphicsWindow(title='Threeway Raster plots')
    mainfig.resize(1200,850)
    subfig1 = mainfig.addPlot(row=0, col=0)
    subfig2 = mainfig.addPlot(row=1, col=0)
    subfig3 = mainfig.addPlot(row=2, col=0)
    subfig1.setXLink(subfig2)
    subfig3.setXLink(subfig1)
    
    plot_A = Rasterplot([TW.monitors['spikemon_A']],
                        neuron_id_range=(0,TW.A.num_neurons),
                        title="Population A",
                        ylabel='Neuron ID', xlabel='time, s',
                        mainfig=mainfig, subfig_rasterplot=subfig1,
                        backend='pyqtgraph', QtApp=app,
                        show_immediately=False)
    
    plot_B = Rasterplot([TW.monitors['spikemon_B']],
                        neuron_id_range=(0,TW.B.num_neurons),
                        title="Population B",
                        ylabel='Neuron ID', xlabel='time, s',
                        mainfig=mainfig, subfig_rasterplot=subfig2,
                        backend='pyqtgraph', QtApp=app,
                        show_immediately=False)
    
    plot_C = Rasterplot([TW.monitors['spikemon_C']],
                        neuron_id_range=(0,TW.C.num_neurons),
                        title="Population C",
                        ylabel='Neuron ID', xlabel='time, s',
                        mainfig=mainfig, subfig_rasterplot=subfig3,
                        backend='pyqtgraph', QtApp=app,
                        show_immediately=True)

    return mainfig


def gaussian(mu, sigma, amplitude, input_size):
    """
        Generate rates based on the gaussian profile
    """
    i = np.arange(input_size)
    shift = mu % 1
    coarse = int(mu - shift)
#    dist = amplitude*np.exp(-np.power((i - shift - int(input_size/2))/input_size, 2.) / (2 * np.power(sigma, 2.)))
    dist = amplitude*np.max([np.exp(-np.power((i - shift -
                            int(input_size/2))/input_size, 2.) /
                            (2 * np.power(sigma, 2.))),
            np.exp(-np.power((i - shift - int(input_size/2))/input_size+1, 2.) /
                       (2 * np.power(sigma, 2.)))], axis = 0)
    return dist[(int(input_size/2) - coarse + i)%input_size]

def double2pop_code(value, input_size, sigma=None, amplitude=100):
    """
        Generate rates based on the gaussian profile
    """
    if sigma is None:
        sigma = 1/input_size
    
    mu = value*input_size % input_size
    activity = gaussian(mu, sigma, amplitude, input_size)
    return activity * Hz


def pop_code2double(pop_array):
    """Calculate circular mean of an array

    @author: Peter Diehl
    """
    size = len(pop_array)
    complex_unit_roots = np.array(
        [np.exp(1j * (2 * np.pi / size) * cur_pos) for cur_pos in range(size)])
    cur_pos = (np.angle(np.sum(pop_array * complex_unit_roots)) %
               (2 * np.pi)) / (2 * np.pi)
    return cur_pos%1


def get_rates(spikemon, measurement_period=100 * ms):
    """
        Get firing rates of neurons based on most recent activity within
        the measurement period
    """
    rates = np.zeros(len(spikemon.event_trains()))
    rates = [len(spikemon.event_trains()[i][spikemon.event_trains()[i] >
                 spikemon.clock.t - measurement_period]) / measurement_period
             for i in range(len(spikemon.event_trains()))]

    #  if debug and len(spikemon.t):
    #      print('Simulation time', spikemon.t / ms, 'ms')
    return rates

if __name__ == '__main__':
    
    prefs.codegen.target = "numpy"
    
    TW = Threeway('TestTW', debug = True)
