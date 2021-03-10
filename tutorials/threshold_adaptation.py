import numpy as np

from brian2 import ms, mV, prefs, SpikeMonitor, StateMonitor, defaultclock,\
        SpikeGeneratorGroup

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.stimuli.testbench import SequenceTestbench
from teili.tools.group_tools import add_group_params_re_init
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

#############
# Load models
syn_model=SynapseEquationBuilder(base_unit='current',
                                 kernel='exponential',
                                 plasticity='non_plastic')
adapt_neu_model=NeuronEquationBuilder(base_unit='voltage',
                                      leak='leaky',
                                      position='spatial',
                                      intrinsic_excitability='threshold_adaptation')
neu_model=NeuronEquationBuilder(base_unit='voltage',
                                leak='leaky',
                                position='spatial')

# Initialize simulation preferences
prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms

# Initialize input sequence: Poisson pattern presented many times
num_inputs = 50
num_items = 1
input_rate = 1
item_duration = 50
pattern_repetitions = 10

repeated_input = SequenceTestbench(num_inputs, num_items, item_duration,
                                   rate=input_rate)
spike_indices, spike_times = repeated_input.stimuli()
poisson_spikes = SpikeGeneratorGroup(num_inputs, spike_indices, spike_times,
                period=50*ms)

sim_time = item_duration * pattern_repetitions

#################
# Building network
num_exc = 20
num_inh = 5
exc_cells = Neurons(num_exc,
                    equation_builder=adapt_neu_model(num_inputs=3),
                    name='exc_cells',
                    verbose=True)
inh_cells = Neurons(num_inh,
                    equation_builder=neu_model(num_inputs=2),
                    name='inh_cells',
                    verbose=True)

feedforward_exc = Connections(poisson_spikes, exc_cells,
                              equation_builder=syn_model(),
                              name='feedforward_exc')
exc_inh_conn = Connections(exc_cells, inh_cells,
                           equation_builder=syn_model(),
                           name='exc_inh_conn')
inh_exc_conn = Connections(inh_cells, exc_cells,
                           equation_builder=syn_model(),
                           name='inh_exc_conn')
exc_exc_conn = Connections(exc_cells, exc_cells,
                           equation_builder=syn_model(),
                           name='exc_exc_conn')
inh_inh_conn = Connections(inh_cells, inh_cells,
                           equation_builder=syn_model(),
                           name='inh_inh_conn')
feedforward_exc.connect(p=.1)
exc_inh_conn.connect()
inh_exc_conn.connect()
exc_exc_conn.connect()
inh_inh_conn.connect(p=.1)

# Parameters
feedforward_exc.tausyn = 5*ms
exc_inh_conn.tausyn = 5*ms
exc_exc_conn.tausyn = 5*ms
inh_exc_conn.tausyn = 10*ms
inh_inh_conn.tausyn = 10*ms
exc_cells.Vm = exc_cells.EL
inh_cells.Vm = inh_cells.EL
inh_exc_conn.weight = -0.1
inh_inh_conn.weight = -0.1
feedforward_exc.weight = 0.1
exc_inh_conn.weight = 0.1
exc_exc_conn.weight = 0.1

exc_cells.thr_min = -69*mV
exc_cells.thr_max = -40*mV

##################
# Setting up monitors
spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
spikemon_inh_neurons = SpikeMonitor(inh_cells, name='spikemon_inh_neurons')
spikemon_seq_neurons = SpikeMonitor(poisson_spikes, name='spikemon_seq_neurons')
rec_neurons = np.random.choice(num_exc, 7, replace=False)
statemon_thresh = StateMonitor(exc_cells, variables=['Vthr'],
                               record=True,
                               name='statemon_thresh')
maoi = StateMonitor(exc_cells, variables=['Iin','Vm'], record=True, name='maoi')

net = TeiliNetwork()
net.add(poisson_spikes, exc_cells, inh_cells, inh_exc_conn, feedforward_exc,
        spikemon_exc_neurons, spikemon_inh_neurons, exc_inh_conn, 
        spikemon_seq_neurons, inh_inh_conn, exc_exc_conn, statemon_thresh,
        maoi)
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

win1 = pg.GraphicsWindow()
RC = Rasterplot(MyEventsModels=[exc_raster],
                backend='pyqtgraph',
                title='Spikes from excitatory neurons',
                ylabel='Indices',
                xlabel='Time (s)',
                QtApp=QtApp,
                mainfig=win1)
win2 = pg.GraphicsWindow()
RC = Rasterplot(MyEventsModels=[inh_raster],
                backend='pyqtgraph',
                title='Spikes from inhibitory neurons',
                ylabel='Indices',
                xlabel='Time (s)',
                QtApp=QtApp,
                mainfig=win2)
win3 = pg.GraphicsWindow()
RC = Rasterplot(MyEventsModels=[seq_raster],
                title='Input spikes',
                ylabel='Indices',
                xlabel='Time (s)',
                backend='pyqtgraph',
                QtApp=QtApp,
                mainfig=win3)
win4 = pg.GraphicsWindow()
LC = Lineplot(DataModel_to_x_and_y_attr=[(thresh_traces, ('t_Vthr', 'Vthr'))],
              #MyPlotSettings=PlotSettings(colors=['r', 'b', 'g', 'c', 'k', 'm', 'y']),
              title='threshold decay',
              xlabel='time (s)',
              ylabel='Vth (V)',
              backend='pyqtgraph',
              QtApp=QtApp,
              mainfig=win4,
              #subgroup_labels=[str(x) for x in rec_neurons],
              show_immediately=True)
