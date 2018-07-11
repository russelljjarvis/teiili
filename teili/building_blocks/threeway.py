#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:09:49 2018

@author: dzenn
"""

import numpy as np
# import matplotlib.pyplot as plt
from pyqtgraph import QtGui, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph as pg
import sys

from brian2 import ms, SpikeMonitor,\
    prefs, PoissonGroup, Hz

from teili.tools.plotting import plot_spikemon_qt

from teili.building_blocks.building_block import BuildingBlock
from teili.building_blocks.wta import WTA
from teili.core.groups import Neurons, Connections
from teili.tools.live import PlotGUI

from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn

from teili.tools.three_way_kernels import A_plus_B_equals_C

prefs.codegen.target = "numpy"
# set_device('cpp_standalone')

# TODO: Add threeway block parameters
threewayParams = {}

wtaParams = {'weInpWTA': 200.8,
             'weWTAInh': 30,
             'wiInhWTA': -30,
             'weWTAWTA': 30,
             'rpWTA': 2.5 * ms,
             'rpInh': 1 * ms,
             'sigm': 2.2
             }

threewayParams.update(wtaParams)


class Threeway(BuildingBlock):
  """A network of three 1d WTA populations connected to a hidden 2d WTA
  population implementing a three-way relation between three 1d quantities 
  A, B and C (e.g. A + B = C) via a hidden population H.

  Attributes:
      Groups (dict): Complete list of keys of neuron groups, synapse groups and
                     WTA substructures of the Threeway building block to be
                     included into Network object for simulation
      Monitors (dict): Complete list of Brian2 monitors for all entities of
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
               block_params=threewayParams,
               num_input_neurons=16,
               num_hidden_neurons=256,
               hidden_layer_gen_func=A_plus_B_equals_C,
               cutoff=5,
               additional_statevars=[],
               spatial_kernel=None,
               monitor=True,
               debug=False):
    """Summary (TBD)

    Args:
        name (str, required): Name of the TW block
        neuron_eq_builder (class, optional): neuron class as imported
                                             from models/neuron_models
        synapse_eq_builder (class, optional): synapse class as imported
                                              from models/synapse_models
        block_params (dict, optional): Parameters for neuron populations
        num_neurons (int, optional): Size of a single neuron population
        fraction_inh_neurons (float, optional): Set to None to skip Dale's priciple
        additional_statevars (list, optional): List of additonal statevariables which are not standard
        num_inputs (int, optional): Number of input currents to R
        monitor (bool, optional): Flag to auto-generate spike and statemonitors
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

    self.Groups, self.Monitors, \
        self.standalone_params = gen_threeway(name,
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

   # Creating handles for neuron groups and inputs
    self.A = self.Groups['wta_A']
    self.B = self.Groups['wta_B']
    self.C = self.Groups['wta_C']
    self.H = self.Groups['wta_H']

    self.Inp_A = self.Groups['inputGroup_A']
    self.Inp_B = self.Groups['inputGroup_B']
    self.Inp_C = self.Groups['inputGroup_C']
    
    self.value_a = np.NAN
    self.value_b = np.NAN
    self.value_c = np.NAN

  def plot(self, start_time=0 * ms, end_time=None):
    """Plot three rasters for input/output populations A, B and C

    Args:
        start_time (int*ms, optional): Start time of plot in ms
        end_time (int*ms, optional): End time of plot in ms
    """

#        if end_time is None:
#            if len(self.spikemonR.t) > 0:
#                end_time = max(self.spikemonR.t)
#            else:
#                end_time = end_time * ms
    tw_raster = plot_threeway_raster(self.Monitors, self.name, start_time, end_time)
    return tw_raster

  def show_parameter_gui(self):
        """Constructs and shows a parameter GUI which allows
        to set A, B and C inputs on-line, or switch them off
        """
      
        param_a = Parameter.create(name = "value A", type = "float",\
                                   value = 0.1, step = 0.05, limits = [0, 1])
        param_b = Parameter.create(name = "value B", type = "float",\
                                   value = 0.1, step = 0.05, limits = [0, 1])
        param_c = Parameter.create(name = "value C", type = "float",\
                                   value = 0.1, step = 0.05, limits = [0, 1])
        param_reset_a = Parameter.create(name = "Shut off A", type = "action")
        param_reset_b = Parameter.create(name = "Shut off B", type = "action")
        param_reset_c = Parameter.create(name = "Shut off C", type = "action")
        twparams = Parameter.create(name = "Three way params", type = "group")
        twparams.addChild(param_a)
        twparams.addChild(param_reset_a)
        twparams.addChild(param_b)
        twparams.addChild(param_reset_b)
        twparams.addChild(param_c)
        twparams.addChild(param_reset_c)
        twparams.param('value A').sigValueChanging.connect(self._param_changed)
        twparams.param('Shut off A').sigActivated.connect(self.reset_A)
        twparams.param('value B').sigValueChanging.connect(self._param_changed)
        twparams.param('Shut off B').sigActivated.connect(self.reset_B)
        twparams.param('value C').sigValueChanging.connect(self._param_changed)
        twparams.param('Shut off C').sigActivated.connect(self.reset_C)
        TwTree = ParameterTree()
        TwTree.setParameters(twparams)
        TwTree.show()
        
        return TwTree

  def plot_live_inputs(self, start_time=0 * ms, end_time=None):
      #TODO: replace the update function with usage of
      #neuron group state variables
      """
      Creates a plotGUI instance and a new timer to run
      the update function (will be replaced with )
      """
            
      app = QtGui.QApplication.instance()
      if app  is None:
          app = QtGui.QApplication(sys.argv)
      else:
          print('QApplication instance already exists: %s' % str(app))
          
          
      if end_time == None:
          if len(self.Monitors['spikemon_A'].t):
              end_time = max(self.Monitors['spikemon_A'].t)
          else:
              end_time = 0*ms
              
      self.measurement_period=end_time - start_time
      
      self.rates_A =  get_rates(self.Monitors['spikemon_A'])
      self.rates_B =  get_rates(self.Monitors['spikemon_B'])
      self.rates_C =  get_rates(self.Monitors['spikemon_C'])
      
      plot_gui = PlotGUI(data = self.rates_A/Hz)
  
      plot_gui.nextRow()
  
      plot_gui.add(data = self.rates_B/Hz)
      plot_gui.nextRow()
      plot_gui.add(data = self.rates_C/Hz)
      
      timer = QtCore.QTimer()
      timer.timeout.connect(self._update_live_inputs)
      timer.start(100)
      
      self.plot_gui = plot_gui
      
      app.exec_()
      
      return plot_gui

  def _update_live_inputs(self):
      """
      Updates curves of the PlotGUI while live plotting the population codes of the populations A, B and C
      """
      
      self.rates_A =  get_rates(self.Monitors['spikemon_A'])
      self.rates_B =  get_rates(self.Monitors['spikemon_B'])
      self.rates_C =  get_rates(self.Monitors['spikemon_C'])
      rates_ABC = [self.rates_A, self.rates_B, self.rates_C]
      for curve, rates in zip(self.plot_gui.curveData, rates_ABC):
          self.plot_gui.curveData[curve] = rates/Hz
#          print(rates/Hz)
#      self.plot_gui.update()
      a, b, c = self.get_values()

      print("A = %g, B = %g, C = %g, t = %g ms" % (a,b,c, self.Monitors['spikemon_A'].clock.t/ ms))
      #print(self.plot_gui.curveData.values())
      #print(rates_ABC)
      
      
  def set_A(self, value):
      """
      Sets spiking rates of neurons of the PoissonGroup Inp_A to satisfy a shape of a gaussian bump centered at 'value' between 0 and 1
      
      Args:
          value (float): a value to be encoded with an activity bump
      """
      
      self.Inp_A.rates = double2pop_code(value, self.num_input_neurons)
      self.value_a = value

  def set_B(self, value):
      """
      Sets spiking rates of neurons of the PoissonGroup Inp_B to satisfy a shape of a gaussian bump centered at 'value' between 0 and 1
      
      Args:
          value (float): a value to be encoded with an activity bump
      """
            
      self.Inp_B.rates = double2pop_code(value, self.num_input_neurons)
      self.value_b = value

  def set_C(self, value):
      """
      Sets spiking rates of neurons of the PoissonGroup Inp_C to satisfy a shape of a gaussian bump centered at 'value' between 0 and 1
      
      Args:
          value (float): a value to be encoded with an activity bump
      """
      
      self.Inp_C.rates = double2pop_code(value, self.num_input_neurons)
      self.value_c = value

  def reset_A(self):
      """
      Resets spiking rates of neurons of the PoissonGroup Inp_A to zero (e.g. turns the input A off)
      """
      self.Inp_A.rates = np.zeros(self.num_input_neurons) * Hz
      self.value_a = np.NAN

  def reset_B(self):
      """
      Resets spiking rates of neurons of the PoissonGroup Inp_B to zero (e.g. turns the input B off)
      """
      self.Inp_B.rates = np.zeros(self.num_input_neurons) * Hz
      self.value_b = np.NAN

  def reset_C(self):
      """
      Resets spiking rates of neurons of the PoissonGroup Inp_A to zero (e.g. turns the input A off)
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
      
  def _param_changed(self, param, value):
      """
      A helper function for live updating the inputs through GUI
      """
      if param.name() == "value A":
          self.set_A(value)
      if param.name() == "value B":
          self.set_B(value)
      if param.name() == "value C":
          self.set_C(value)

  

  def get_values(self, measurement_period=100 * ms):
      """
      Extracts encoded values of A, B and C from the spiking rates of
      the corresponding populations
      
      Args:
          measurement_period (ms, optional): Sets the interval back from
          current moment in time for the spikes to be included into rate calculation
      """
      
      if self.A.monitor == True and self.B.monitor == True and self.C.monitor == True:
        a = pop_code2double(get_rates(self.A.spikemonWTA,
                                      measurement_period=measurement_period))
        b = pop_code2double(get_rates(self.B.spikemonWTA,
                                      measurement_period=measurement_period))
        c = pop_code2double(get_rates(self.C.spikemonWTA,
                                      measurement_period=measurement_period))
        return a, b, c
      else:
        raise ValueError(
            'Unable to compute population vectors, monitoring has been turned off!')


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
  # TODO: Check to have a name
  

  wtaParams = {'weInpWTA': block_params['weInpWTA'],
               'weWTAInh': block_params['weWTAInh'],
               'wiInhWTA': block_params['wiInhWTA'],
               'weWTAWTA': block_params['weWTAWTA'],
               'rpWTA': block_params['rpWTA'],
               'rpInh': block_params['rpInh'],
               'sigm': block_params['sigm']
               }
  if debug:
    print("Creating WTA's!")

  wta_A = WTA('wta_A', dimensions=1, block_params=wtaParams, num_inputs=2, num_neurons=num_input_neurons,
              num_inh_neurons=int(0.2 * num_input_neurons), cutoff=cutoff, monitor=True, debug=debug)
  wta_B = WTA('wta_B', dimensions=1, block_params=wtaParams, num_inputs=2, num_neurons=num_input_neurons,
              num_inh_neurons=int(0.2 * num_input_neurons), cutoff=cutoff, monitor=True, debug=debug)
  wta_C = WTA('wta_C', dimensions=1, block_params=wtaParams, num_inputs=2, num_neurons=num_input_neurons,
              num_inh_neurons=int(0.2 * num_input_neurons), cutoff=cutoff, monitor=True, debug=debug)
  wta_H = WTA('wta_H', dimensions=2, block_params=wtaParams, num_inputs=3, num_neurons=num_input_neurons,
              num_inh_neurons=int(0.2 * num_hidden_neurons), cutoff=cutoff, monitor=monitor, debug=debug)

  Groups = {
      'wta_A': wta_A,
      'wta_B': wta_B,
      'wta_C': wta_C,
      'wta_H': wta_H}

  inputGroup_A = PoissonGroup(
      num_input_neurons, rates=np.zeros(num_input_neurons) * Hz)
  inputGroup_B = PoissonGroup(
      num_input_neurons, rates=np.zeros(num_input_neurons) * Hz)
  inputGroup_C = PoissonGroup(
      num_input_neurons, rates=np.zeros(num_input_neurons) * Hz)

  input_groups = {
      'inputGroup_A': inputGroup_A,
      'inputGroup_B': inputGroup_B,
      'inputGroup_C': inputGroup_C}

  # Creating interpopulation synapse groups
  synAH1e = Connections(wta_A.group, wta_H.group, equation_builder=synapse_eq_builder(),
                        method="euler", name='s' + name + '_A_to_H')
  synHA1e = Connections(wta_H.group, wta_A.group, equation_builder=synapse_eq_builder(),
                        method="euler", name='s' + name + '_H_to_A')
  synBH1e = Connections(wta_B.group, wta_H.group, equation_builder=synapse_eq_builder(),
                        method="euler", name='s' + name + '_B_to_H')
  synHB1e = Connections(wta_H.group, wta_B.group, equation_builder=synapse_eq_builder(),
                        method="euler", name='s' + name + '_H_to_B')
  synCH1e = Connections(wta_C.group, wta_H.group, equation_builder=synapse_eq_builder(),
                        method="euler", name='s' + name + '_C_to_H')
  synHC1e = Connections(wta_H.group, wta_C.group, equation_builder=synapse_eq_builder(),
                        method="euler", name='s' + name + '_H_to_C')

  # Creating input synapse groups
  synInpA1e = Connections(inputGroup_A, wta_A.group, equation_builder=synapse_eq_builder(),
                          method="euler", name='s' + name + '_Inp_to_A')
  synInpB1e = Connections(inputGroup_B, wta_B.group, equation_builder=synapse_eq_builder(),
                          method="euler", name='s' + name + '_Inp_to_B')
  synInpC1e = Connections(inputGroup_C, wta_C.group, equation_builder=synapse_eq_builder(),
                          method="euler", name='s' + name + '_Inp_to_C')

  interPopSynGroups = {
      'synAH1e': synAH1e,
      'synHA1e': synHA1e,
      'synBH1e': synBH1e,
      'synHB1e': synHB1e,
      'synCH1e': synCH1e,
      'synHC1e': synHC1e}

  synGroups = {
      'synInpA1e': synInpA1e,
      'synInpB1e': synInpB1e,
      'synInpC1e': synInpC1e}

  for tmp_syn_group in synGroups:
    synGroups[tmp_syn_group].connect('i == j')
    synGroups[tmp_syn_group].weight = wtaParams['weInpWTA']

  synGroups.update(interPopSynGroups)

  # Connecting the populations with a given index generation function
  # TODO: add more index functions
  index_gen_function = hidden_layer_gen_func

  for tmp_syn_group in interPopSynGroups:
    arr_i, arr_j = index_gen_function(
        tmp_syn_group[3], tmp_syn_group[4], num_input_neurons)
    interPopSynGroups[tmp_syn_group].connect(i=arr_i, j=arr_j)
    interPopSynGroups[tmp_syn_group].weight = wtaParams['weInpWTA']

  Groups.update(input_groups)
  Groups.update(synGroups)

  if monitor:
    spikemon_InpA = SpikeMonitor(
        inputGroup_A, name='spikemon' + name + '_InpA')
    spikemon_InpB = SpikeMonitor(
        inputGroup_B, name='spikemon' + name + '_InpB')
    spikemon_InpC = SpikeMonitor(
        inputGroup_C, name='spikemon' + name + '_InpC')

  Monitors = {
      'spikemon_InpA': spikemon_InpA,
      'spikemon_InpB': spikemon_InpB,
      'spikemon_InpC': spikemon_InpC,
      'spikemon_A': wta_A.Monitors['spikemonWTA'],
      'spikemon_B': wta_B.Monitors['spikemonWTA'],
      'spikemon_C': wta_C.Monitors['spikemonWTA']}

  #Monitors.update(wta_A.Monitors, wta_B.Monitors, wta_C.Monitors, wta_H.Monitors)

  standalone_params = {}

  return Groups, Monitors, standalone_params


def plot_threeway_raster(tw_monitors, name, start_time, end_time):
  """Function to easily visualize WTA activity.

  Args:
      name (str, required): Name of the WTA population
      start_time (brian2.units.fundamentalunits.Quantity, required): Start time in ms
          from when network activity should be plotted.
      end_time (brian2.units.fundamentalunits.Quantity, required): End time in ms of plot.
          Can be smaller than simulation time but not larger
      wta_monitors (dict.): Dictionary with keys to access spike- and statemonitors. in WTA.Monitors
  """
  app = QtGui.QApplication.instance()
  if app is None:
    app = QtGui.QApplication(sys.argv)
  else:
    print('QApplication instance already exists: %s' % str(app))

  pg.setConfigOptions(antialias=True)

  win_raster = pg.GraphicsWindow(
      title='Threeway Relation Network Test Simulation: Raster plots')
  win_raster.resize(1000, 1800)
  win_raster.setWindowTitle(
      'Threeway Relation Network Test Simulation: Raster plots')

  raster_A = win_raster.addPlot(title="Population A")
  win_raster.nextRow()
  raster_B = win_raster.addPlot(title="Population B")
  win_raster.nextRow()
  raster_C = win_raster.addPlot(title="Population C")

  plot_spikemon_qt(monitor=tw_monitors['spikemon_A'], start_time=start_time, end_time=end_time,
                   num_neurons=tw_monitors['spikemon_A'].source.N,
                   window=raster_A)
  plot_spikemon_qt(monitor=tw_monitors['spikemon_B'], start_time=start_time, end_time=end_time,
                   num_neurons=tw_monitors['spikemon_B'].source.N,
                   window=raster_B)
  plot_spikemon_qt(monitor=tw_monitors['spikemon_C'], start_time=start_time, end_time=end_time,
                   num_neurons=tw_monitors['spikemon_C'].source.N,
                   window=raster_C)

  app.exec()

  return win_raster




def gaussian(x, gaussianPeak=100, gaussianSigma=0.1):
  return gaussianPeak * (np.exp(-0.5 * (x / gaussianSigma)**2))


def double2pop_code(value, inputSize):
  """Get firing rates given the position of the gaussian activation bump

  @author: Peter Diehl
  """
  activationFunction = gaussian
  centerID = int(value * inputSize)
  topoCoords = {}
  for i in range(inputSize):
    pos = 1. * float(i) / inputSize
    topoCoords[i] = (0.5, pos)
  center_coords = topoCoords[centerID]
  dists = np.zeros(inputSize)

  for i in range(inputSize):
    coords = topoCoords[i]
    deltaX = abs(coords[0] - center_coords[0])
    deltaY = abs(coords[1] - center_coords[1])
    if deltaX > 0.5:
      deltaX = 1.0 - deltaX
    if deltaY > 0.5:
      deltaY = 1.0 - deltaY
    squared_dist = deltaX ** 2 + deltaY ** 2
    dists[i] = squared_dist
  distsAndIds = zip(dists, range(inputSize))
  distsAndIds = sorted(distsAndIds)
  unused_sorted_dists, dist_sorted_ids = zip(*distsAndIds)
  activity = np.zeros(inputSize)
  for i, idx in enumerate(dist_sorted_ids):
    activity[idx] = activationFunction(float(i) / inputSize)
  return activity * Hz


def pop_code2double(popArray):
  """Calculate circular mean of an array

  @author: Peter Diehl
  """
  size = len(popArray)
  complex_unit_roots = np.array(
      [np.exp(1j * (2 * np.pi / size) * cur_pos) for cur_pos in range(size)])
  cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) %
             (2 * np.pi)) / (2 * np.pi)
  return cur_pos


def get_rates(spikeMon, measurement_period=100 * ms):
  """Get firing rates of neurons based on most recent activity within the measurement period

  """
  rates = np.zeros(len(spikeMon.event_trains())) 
  rates = [len(spikeMon.event_trains()[i][spikeMon.event_trains()[i] > spikeMon.clock.t - measurement_period]) / measurement_period
       for i in range(len(spikeMon.event_trains()))]
  
#  if debug and len(spikeMon.t):
#      print('Simulation time', spikeMon.t / ms, 'ms')
  return rates
