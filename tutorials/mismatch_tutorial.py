#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a tutorial that shows how to add mismatch to a SNN.
As an example, here we connect one input neuron to 1000 output neurons.
(Neuron model: DPI, Synapse model: DPISyn).
The standard deviation of the mismatch distribution is specified in the
dictionaries:
    - mismatch_neuron_param
    - mismatch_synap_param
Here mismatch is added to the neuron refractory period (refP) and to the synaptic
weight (baseweight).

Created on Wed Jul 25 18:32:44 2018
@author: nrisi
"""
import os
import numpy as np
import sys

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from brian2 import SpikeGeneratorGroup, SpikeMonitor, StateMonitor, second, ms, asarray, nA, prefs, set_device
from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import DPI as neuron_model
from teili.models.synapse_models import DPISyn as syn_model

from teili.tools.visualizer.DataModels.StateVariablesModel import StateVariablesModel
from teili.tools.visualizer.DataControllers.Rasterplot import Rasterplot
from teili.tools.visualizer.DataControllers.Lineplot import Lineplot
from teili.tools.visualizer.DataControllers.Histogram import Histogram
from teili.tools.visualizer.DataViewers import PlotSettings

standalone = False
if standalone:
    standalone_dir = os.path.expanduser('~/mismatch_standalone')
    set_device('cpp_standalone', directory=standalone_dir)
prefs.codegen.target = "numpy"

app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))


Net = TeiliNetwork()

mismatch_neuron_param = {
    'Inoise': 0,
    'Iconst': 0,
    'kn': 0,
    'kp': 0,
    'Ut': 0,
    'Io': 0,
    'Cmem': 0,
    'Iath': 0,
    'Iagain': 0,
    'Ianorm': 0,
    'Ica': 0,
    'Itauahp': 0,
    'Ithahp': 0,
    'Cahp': 0,
    'Ishunt': 0,
    'Ispkthr': 0,
    'Ireset': 0,
    'Ith': 0,
    'Itau': 0,
    'refP': 0.2,
}

mismatch_synap_param = {
    'Io_syn': 0,
    'kn_syn': 0,
    'kp_syn': 0,
    'Ut_syn': 0,
    'Csyn': 0,
    'I_tau': 0,
    'I_th': 0,
    'I_syn': 0,
    'w_plast': 0,
    'baseweight': 0.2
}

# Input layer
ts_input = asarray([1, 3, 4, 5, 6, 7, 8, 9]) * ms
ids_input = asarray([0, 0, 0, 0, 0, 0, 0, 0])
input_spikegen = SpikeGeneratorGroup(1, indices=ids_input,
                                     times=ts_input, name='gtestInp')

# Output layer
output_neurons = Neurons(1000, equation_builder=neuron_model(num_inputs=2),
                         name='output_neurons')
output_neurons.refP = 3 * ms
output_neurons.Iconst = 10 * nA

# Input Synapse
input_syn = Connections(input_spikegen, output_neurons, equation_builder=syn_model(),
                        name="inSyn", verbose=False)
input_syn.connect(True)
input_syn.weight = 5

"""Adding mismatch to neurons and synaptic weights:
getattr(output_neurons, mism_param_neu) returns an array of length equal to the
number of neurons. Assuming that mismatch has not been added yet (e.g. if you
have just created the neuron population), the values of the selected parameter
will be the same for all the neurons in the population. (we choose to store
the first one)
"""
if not standalone:
    mean_neuron_param = np.copy(getattr(output_neurons, 'refP'))[0]
    mean_synapse_param = np.copy(getattr(input_syn, 'baseweight'))[0]
else:
    mean_neuron_param = output_neurons.get_params()['refP'][0]
    mean_synapse_param = input_syn._init_parameters['baseweight']
    
output_neurons.add_mismatch(std_dict=mismatch_neuron_param, seed=10)
input_syn.add_mismatch(std_dict=mismatch_synap_param, seed=11)

# %%
# Setting monitors:
spikemon_input = SpikeMonitor(input_spikegen, name='spikemon_input')
spikemon_output = SpikeMonitor(output_neurons, name='spikemon_output')
statemon_output = StateMonitor(output_neurons,
                               variables=['Imem'],
                               record=range(0,output_neurons.N),
                               name='statemonNeuMid')
statemon_input_syn = StateMonitor(input_syn,
                                  variables='I_syn',
                                  record=range(0,output_neurons.N),
                                  name='statemon_input_syn')

Net.add(input_spikegen, output_neurons, input_syn,
        spikemon_input, spikemon_output,
        statemon_output, statemon_input_syn)

# Run simulation for 500 ms
duration = .500
Net.run(duration * second)


# START PLOTTING
# define general settings
pg.setConfigOptions(antialias=True)
MyPlotSettings = PlotSettings(fontsize_title=12,
                              fontsize_legend=12,
                              fontsize_axis_labels=12,
                              marker_size=2)

# prepare data (part 1)
neuron_ids_to_plot = np.random.randint(1000, size=5)

distinguish_neurons_in_plot = True  # show values in different color per neuron otherwise the same color per subgroup

## plot EPSC (subfig3)
if distinguish_neurons_in_plot:
    # to get every neuron plotted with a different color to distinguish them
    DataModels_EPSC = []
    for neuron_id in neuron_ids_to_plot:
        MyData_EPSC = StateVariablesModel(state_variable_names=['EPSC'],
                                          state_variables=[statemon_input_syn.I_syn[neuron_id]],
                                          state_variables_times=[statemon_input_syn.t])
        DataModels_EPSC.append((MyData_EPSC, ('t_EPSC', 'EPSC')))
else:
    # to get all neurons plotted in the same color
    neuron_ids_to_plot = np.random.randint(1000, size=5)
    MyData_EPSC = StateVariablesModel(state_variable_names=['EPSC'],
                                 state_variables=[statemon_input_syn.I_syn[neuron_ids_to_plot].T],
                                 state_variables_times=[statemon_input_syn.t])
    DataModels_EPSC=[(MyData_EPSC, ('t_EPSC', 'EPSC'))]

## plot Imem (subfig4)
if distinguish_neurons_in_plot:
    # to get every neuron plotted with a different color to distinguish them
    DataModels_Imem = []
    for neuron_id in neuron_ids_to_plot:
        MyData_Imem = StateVariablesModel(state_variable_names=['Imem'],
                                          state_variables=[statemon_output.Imem[neuron_id].T],
                                          state_variables_times=[statemon_output.t])
        DataModels_Imem.append((MyData_Imem, ('t_Imem', 'Imem')))
else:
    # to get all neurons plotted in the same color
    neuron_ids_to_plot = np.random.randint(1000, size=5)
    MyData_Imem = StateVariablesModel(state_variable_names=['Imem'],
                                      state_variables=[statemon_output.Imem[neuron_ids_to_plot].T],
                                      state_variables_times=[statemon_output.t])
    DataModels_Imem=[(MyData_Imem, ('t_Imem', 'Imem'))]


# set up main window and subplots (part 1)
QtApp = QtGui.QApplication([])
mainfig = pg.GraphicsWindow(title='Simple SNN')
subfig1 = mainfig.addPlot(row=0, col=0)
subfig2 = mainfig.addPlot(row=1, col=0)
subfig3 = mainfig.addPlot(row=2, col=0)
subfig4 = mainfig.addPlot(row=3, col=0)

# add data to plots
Rasterplot(MyEventsModels=[spikemon_input],
                      MyPlotSettings=MyPlotSettings,
                      time_range=[0, duration],
                      title="Spike generator", xlabel="Time (ms)", ylabel="Neuron ID",
                      backend='pyqtgraph', mainfig=mainfig, subfig_rasterplot=subfig1, QtApp=QtApp,
                      show_immediately=False)
Rasterplot(MyEventsModels=[spikemon_output],
                     MyPlotSettings=MyPlotSettings,
                     time_range=[0, duration],
                     title="Output layer", xlabel="Time (ms)", ylabel="Neuron ID",
                     backend='pyqtgraph', mainfig=mainfig, subfig_rasterplot=subfig2, QtApp=QtApp,
                     show_immediately=False)
Lineplot(DataModel_to_x_and_y_attr=DataModels_EPSC,
                   MyPlotSettings=MyPlotSettings,
                   x_range=[0, duration],
                   title="EPSC", xlabel="Time (ms)", ylabel="EPSC (pA)",
                   backend='pyqtgraph', mainfig=mainfig, subfig=subfig3, QtApp=QtApp,
                   show_immediately=False)
Lineplot(DataModel_to_x_and_y_attr=DataModels_Imem,
                   MyPlotSettings=MyPlotSettings,
                   x_range=[0, duration],
                   title="I_mem", xlabel="Time (ms)", ylabel="Membrane current Imem (nA)",
                   backend='pyqtgraph', mainfig=mainfig, subfig=subfig4, QtApp=QtApp,
                   show_immediately=True)


# prepare data (part 1)
input_syn_baseweights = np.asarray(getattr(input_syn, 'baseweight'))*10**12
MyData_baseweight = StateVariablesModel(state_variable_names=['baseweight'],
                                          state_variables=[input_syn_baseweights])  # to pA

refractory_periods = np.asarray(getattr(output_neurons, 'refP'))*10**3 # to ms
MyData_refP = StateVariablesModel(state_variable_names=['refP'],
                                  state_variables=[refractory_periods])

# set up main window and subplots (part 2)
mainfig = pg.GraphicsWindow(title='Mismatch distribution')
subfig1 = mainfig.addPlot(row=0, col=0)
subfig2 = mainfig.addPlot(row=1, col=0)

# add data to plots
Histogram(DataModel_to_attr=[(MyData_baseweight, 'baseweight')],
                    MyPlotSettings=MyPlotSettings,
                    title='baseweight', xlabel='(pA)', ylabel='count',
                    backend='pyqtgraph',
                    mainfig=mainfig, subfig=subfig1, QtApp=QtApp,
                    show_immediately=False)
y, x = np.histogram(input_syn_baseweights, bins="auto")
subfig1.plot(x=np.asarray([mean_synapse_param*10**12, mean_synapse_param*10**12]),
             y=np.asarray([0, 300]),
                pen=pg.mkPen((0, 255, 0), width=2))

Histogram(DataModel_to_attr=[(MyData_refP, 'refP')],
                    MyPlotSettings=MyPlotSettings,
                    title='refP', xlabel='(ms)', ylabel='count',
                    backend='pyqtgraph',
                    mainfig=mainfig, subfig=subfig2, QtApp=QtApp,
                    show_immediately=False)
subfig2.plot(x=np.asarray([mean_neuron_param*10**3, mean_neuron_param*10**3]),
             y=np.asarray([0, 450]),
        pen=pg.mkPen((0, 255, 0), width=2))

app.exec()