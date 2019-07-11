
# -*- coding: utf-8 -*-
# @Author: kburel
# @Date:   2018-26-11 12:34:16
# @Last Modified by:   Karla Burelo
# -*- coding: utf-8 -*-

"""
This is a tutorial example used to learn how the different synaptic kernels behave and 
how to set them in your network
"""

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

from brian2 import second, ms, prefs,\
        SpikeMonitor, StateMonitor,\
        SpikeGeneratorGroup

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import DPI as neuron_model
from teili.models.synapse_models import Alpha, Resonant
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import  StateVariablesModel
from teili.tools.visualizer.DataControllers import Lineplot


prefs.codegen.target = "numpy"

input_timestamps = np.asarray([1, 1.5, 1.8, 2.0, 2.0, 2.3, 2.5, 3]) * ms
input_indices = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])

input_spikegenerator = SpikeGeneratorGroup(2, indices=input_indices,
                                times=input_timestamps, name='gtestInp')

Net = TeiliNetwork()

testNeurons = Neurons(1, equation_builder=neuron_model(num_inputs=2), name="testNeuron")
testNeurons.set_params(neuron_model_param)
testNeurons.refP = 1 * ms

testNeurons2 = Neurons(1, equation_builder=neuron_model(num_inputs=2), name="testNeuron2")
testNeurons2.set_params(neuron_model_param)
testNeurons2.refP = 1 * ms

#Set synapses using different kernels
InpSynAlpha = Connections(input_spikegenerator, testNeurons,
                     equation_builder=Alpha(), name="testSynAlpha", verbose=False)
InpSynAlpha.connect(True)
InpSynAlpha.weight = np.asarray([10,-10])

InpSynResonant = Connections(input_spikegenerator, testNeurons2,
                     equation_builder=Resonant(), name="testSynResonant", verbose=False)
InpSynResonant.connect(True)
InpSynResonant.weight = np.asarray([10,-10])


#Set monitors
spikemonInp = SpikeMonitor(input_spikegenerator, name='spikemonInp')
statemonInpSynAlpha = StateMonitor(
    InpSynAlpha, variables='I_syn', record=True, name='statemonInpSynAlpha')
statemonInpSynResonant = StateMonitor(
    InpSynResonant, variables='I_syn', record=True, name='statemonInpSynResonant')
statemonNeuOut = StateMonitor(testNeurons, variables=[
                              'Iin'], record=0, name='statemonNeuOut')
statemonNeuOut2 = StateMonitor(testNeurons2, variables=[
                              'Iin'], record=0, name='statemonNeuOut2')

Net.add(input_spikegenerator, testNeurons, testNeurons2,
        InpSynAlpha, InpSynResonant, spikemonInp,
        statemonInpSynAlpha, statemonInpSynResonant,
        statemonNeuOut, statemonNeuOut2)

duration = 0.010
Net.run(duration * second)


#Visualize simulation results
app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))

pg.setConfigOptions(antialias=True)
labelStyle = {'color': '#FFF', 'font-size': 12}
MyPlotSettings = PlotSettings(fontsize_title=labelStyle['font-size'],
                              fontsize_legend=labelStyle['font-size'],
                              fontsize_axis_labels=10,
                              marker_size=7)

win = pg.GraphicsWindow(title='Kernels Simulation')
win.resize(900, 600)
win.setWindowTitle('Simple SNN')

p1 = win.addPlot()
p2 = win.addPlot()
win.nextRow()
p3 = win.addPlot()
p4 = win.addPlot()

# Alpha kernel synapse
data = statemonInpSynAlpha.I_syn.T
data[:, 1] *= -1.
datamodel_SynAlpha = StateVariablesModel(state_variable_names=['I_syn'],
                                state_variables=[data],
                                state_variables_times=[statemonInpSynAlpha.t])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel_SynAlpha, ('t_I_syn', 'I_syn'))],
         MyPlotSettings=MyPlotSettings,
         x_range=(0, duration),
         y_range=None,
         title='Alpha Kernel Synapse',
         xlabel='Time (s)',
         ylabel='Synaptic current I (A)',
         backend='pyqtgraph',
         mainfig=win,
         subfig=p1,
         QtApp=app)
for i, data in enumerate(np.asarray(spikemonInp.t)):
    vLine = pg.InfiniteLine(pen=pg.mkPen(color=(200, 200, 255),
                style=QtCore.Qt.DotLine),pos=data, angle=90, movable=False,)
    p1.addItem(vLine, ignoreBounds=True)

# Neuron response
Lineplot(DataModel_to_x_and_y_attr=[(statemonNeuOut, ('t', 'Iin'))],
         MyPlotSettings=MyPlotSettings,
         x_range=(0, duration),
         y_range=None,
         title='Neuron response',
         xlabel='Time (s)',
         ylabel='Membrane current I_mem (A)',
         backend='pyqtgraph',
         mainfig=win,
         subfig=p2,
         QtApp=app)

# Resonant kernel synapse
data = statemonInpSynResonant.I_syn.T
data[:, 1] *= -1.
datamodel_SynResonant = StateVariablesModel(state_variable_names=['I_syn'],
                                state_variables=[data],
                                state_variables_times=[statemonInpSynResonant.t])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel_SynResonant, ('t_I_syn','I_syn'))],
         MyPlotSettings=MyPlotSettings,
         x_range=(0, duration),
         y_range=None,
         title='Resonant Kernel Synapse',
         xlabel='Time (s)',
         ylabel='Synaptic current I (A)',
         backend='pyqtgraph',
         mainfig=win,
         subfig=p3,
         QtApp=app)
for i, data in enumerate(np.asarray(spikemonInp.t)):
    vLine = pg.InfiniteLine(pen=pg.mkPen(color=(200, 200, 255),
                style=QtCore.Qt.DotLine),pos=data, angle=90, movable=False,)
    p3.addItem(vLine, ignoreBounds=True)

# Neuron response
Lineplot(DataModel_to_x_and_y_attr=[(statemonNeuOut2, ('t', 'Iin'))],
         MyPlotSettings=MyPlotSettings,
         x_range=(0, duration),
         y_range=None,
         title='Neuron response',
         xlabel='Time (s)',
         ylabel='Membrane current I_mem (A)',
         backend='pyqtgraph',
         mainfig=win,
         subfig=p4,
         QtApp=app,
         show_immediately=True)
