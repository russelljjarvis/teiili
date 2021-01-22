import numpy as np
from scipy.stats import norm

from brian2 import Hz, ms, mV, prefs, SpikeMonitor, StateMonitor, defaultclock,\
    ExplicitStateUpdater, SpikeGeneratorGroup, TimedArray, PoissonGroup

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import StochasticLIF as neuron_model
from teili.models.synapse_models import StochasticSyn_decay as static_synapse_model
from teili.stimuli.testbench import SequenceTestbench
from teili.tools.add_run_reg import add_lfsr
from teili.tools.group_tools import add_group_activity_proxy
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.tools.converter import delete_doublets

from lfsr import create_lfsr

import sys
import pickle
import os
from datetime import datetime

#############
# Load models
path = os.path.expanduser("/home/pablo/git/teili")
model_path = os.path.join(path, "teili", "models", "equations", "")
neuron_model_Adapt = NeuronEquationBuilder.import_eq(
        model_path + 'StochLIFAdapt.py')

# Initialize simulation preferences
prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms
stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')
net = TeiliNetwork()

# Initialize input sequence
num_items = 1
num_channels = 50
sequence_duration = 50
noise_prob = None
item_rate = 30
spike_times, spike_indices = [], []
sequence_repetitions = 10
sequence = SequenceTestbench(num_channels, num_items, sequence_duration,
                                     noise_prob, item_rate)
tmp_i, tmp_t = sequence.stimuli()
poisson_spikes = SpikeGeneratorGroup(num_channels, tmp_i, tmp_t,
                period=sequence_duration*ms)

#################
# Building network
# cells
num_exc = 20
num_inh = 5
exc_cells = Neurons(num_exc,
                    equation_builder=neuron_model_Adapt(num_inputs=3),
                    method=stochastic_decay,
                    name='exc_cells',
                    verbose=True)
inh_cells = Neurons(num_inh,
                    equation_builder=neuron_model(num_inputs=2),
                    method=stochastic_decay,
                    name='inh_cells',
                    verbose=True)
# Register proxy arrays
dummy_unit = 1*mV
exc_cells.variables.add_array('activity_proxy', 
                               size=exc_cells.N,
                               dimensions=dummy_unit.dim)
exc_cells.variables.add_array('normalized_activity_proxy', 
                               size=exc_cells.N)

# Connections
feedforward_exc = Connections(poisson_spikes, exc_cells,
                              equation_builder=static_synapse_model(),
                              method=stochastic_decay,
                              name='feedforward_exc')
exc_inh_conn = Connections(exc_cells, inh_cells,
                           equation_builder=static_synapse_model(),
                           method=stochastic_decay,
                           name='exc_inh_conn')
inh_exc_conn = Connections(inh_cells, exc_cells,
                           equation_builder=static_synapse_model(),
                           method=stochastic_decay,
                           name='inh_exc_conn')
exc_exc_conn = Connections(exc_cells, exc_cells,
                           equation_builder=static_synapse_model(),
                           method=stochastic_decay,
                           name='exc_exc_conn')
inh_inh_conn = Connections(inh_cells, inh_cells,
                           equation_builder=static_synapse_model(),
                           method=stochastic_decay,
                           name='inh_inh_conn')
feedforward_exc.connect(p=.1)
exc_inh_conn.connect()
inh_exc_conn.connect()
exc_exc_conn.connect()
inh_inh_conn.connect(p=.1)

# Time constants
feedforward_exc.tau_syn = 5*ms
inh_exc_conn.tau_syn = 10*ms
exc_inh_conn.tau_syn = 5*ms
exc_exc_conn.tau_syn = 5*ms
inh_inh_conn.tau_syn = 10*ms
exc_cells.tau = 19*ms
inh_cells.tau = 10*ms

# LFSR lengths
feedforward_exc.lfsr_num_bits_syn = 5
exc_cells.lfsr_num_bits = 5
inh_cells.lfsr_num_bits = 5
inh_exc_conn.lfsr_num_bits_syn = 5
exc_inh_conn.lfsr_num_bits_syn = 4
exc_exc_conn.lfsr_num_bits_syn = 5
inh_inh_conn.lfsr_num_bits_syn = 5

seed = 12
exc_cells.Vm = 3*mV
inh_cells.Vm = 3*mV

inh_exc_conn.weight = -1
inh_exc_conn.w_plast = 1
inh_inh_conn.weight = -1
feedforward_exc.weight = 5
exc_inh_conn.weight = 5
exc_exc_conn.weight = 1

# Set LFSRs for each group
ta = create_lfsr([exc_cells, inh_cells],
                 [inh_exc_conn, feedforward_exc,
                     exc_inh_conn, exc_exc_conn,
                     inh_inh_conn],
                 defaultclock.dt)

##################
# Setting up monitors
spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
spikemon_inh_neurons = SpikeMonitor(inh_cells, name='spikemon_inh_neurons')
spikemon_seq_neurons = SpikeMonitor(poisson_spikes, name='spikemon_seq_neurons')
rec_neurons = np.random.choice(num_exc, 7, replace=False)
statemon_thresh = StateMonitor(exc_cells, variables=['Vthres'],
                               record=rec_neurons,
                               name='statemon_thresh')

net = TeiliNetwork()
net.add(poisson_spikes, exc_cells, inh_cells, inh_exc_conn, feedforward_exc,
        spikemon_exc_neurons, spikemon_inh_neurons, exc_inh_conn, 
        spikemon_seq_neurons, inh_inh_conn, exc_exc_conn, statemon_thresh)
net.run(10000*ms, report='stdout', report_period=100*ms)

# Plots
from teili.tools.visualizer.DataModels import EventsModel, StateVariablesModel
from teili.tools.visualizer.DataControllers import Rasterplot, Lineplot
from teili.tools.visualizer.DataViewers import PlotSettings
import pyqtgraph as pg
from PyQt5 import QtGui

QtApp = QtGui.QApplication([])

exc_raster = EventsModel.from_brian_spike_monitor(spikemon_exc_neurons)
inh_raster = EventsModel.from_brian_spike_monitor(spikemon_inh_neurons)
seq_raster = EventsModel.from_brian_spike_monitor(spikemon_seq_neurons)
skip_not_rec_neuron_ids = True
thresh_traces = StateVariablesModel.from_brian_state_monitors([statemon_thresh],
        skip_not_rec_neuron_ids)

RC = Rasterplot(MyEventsModels=[exc_raster], backend='pyqtgraph', QtApp=QtApp)
RC = Rasterplot(MyEventsModels=[inh_raster], backend='pyqtgraph', QtApp=QtApp)
RC = Rasterplot(MyEventsModels=[seq_raster], backend='pyqtgraph', QtApp=QtApp)
LC = Lineplot(DataModel_to_x_and_y_attr=[(thresh_traces, ('t_Vthres', 'Vthres'))],
              #MyPlotSettings=PlotSettings(colors=['r', 'b', 'g', 'c', 'k', 'm', 'y']),
              backend='pyqtgraph', QtApp=QtApp,
              subgroup_labels=[str(x) for x in rec_neurons],
              show_immediately=True)
