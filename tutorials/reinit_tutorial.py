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
                                 plasticity='stdp',
                                 structural_plasticity='deterministic_counter',
                                 verbose=True)
neu_model=NeuronEquationBuilder(base_unit='voltage',
                                leak='leaky',
                                position='spatial',
                                verbose=True)

# Initialize simulation preferences
prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms

# Initialize input sequence: Poisson pattern presented many times
num_inputs = 50
num_items = 1
input_rate = 5
item_duration = 50
pattern_repetitions = 200
period = 50*ms

repeated_input = SequenceTestbench(num_inputs, num_items, item_duration,
                                   rate=input_rate)
spike_indices, spike_times = repeated_input.stimuli()
poisson_spikes = SpikeGeneratorGroup(num_inputs, spike_indices, spike_times,
                period=period)

sim_time = item_duration * pattern_repetitions

#################
# Building network
# cells
num_exc = 5
exc_cells = Neurons(num_exc,
                    equation_builder=neu_model(num_inputs=3),
                    name='exc_cells',
                    verbose=True)

# Connections
feedforward_exc = Connections(poisson_spikes, exc_cells,
                              equation_builder=syn_model(),
                              name='feedforward_exc')
feedforward_exc.connect()
# Set sparsity
feedforward_exc.weight = 1
for neu in range(num_exc):
    ffe_zero_w = np.random.choice(num_inputs, int(num_inputs*.7), replace=False)
    feedforward_exc.weight[ffe_zero_w, neu] = 0
    feedforward_exc.w_plast[ffe_zero_w, neu] = 0

# Initializations
feedforward_exc.tausyn = 5*ms
exc_cells.Vm = exc_cells.EL

##################
# Synaptic homeostasis
re_init_dt = 1000*ms
add_group_params_re_init(groups=[feedforward_exc],
                         variable='w_plast',
                         re_init_variable='re_init_counter',
                         re_init_threshold=10,
                         re_init_dt=re_init_dt,
                         dist_param=1,
                         scale=.05,
                         distribution='gamma',
                         clip_min=0,
                         clip_max=1,
                         reference='synapse_counter')
add_group_params_re_init(groups=[feedforward_exc],
                         variable='weight',
                         re_init_variable='re_init_counter',
                         re_init_threshold=10,
                         re_init_dt=re_init_dt,
                         distribution='deterministic',
                         const_value=1,
                         reference='synapse_counter')
add_group_params_re_init(groups=[feedforward_exc],
                         variable='tausyn',
                         re_init_variable='re_init_counter',
                         re_init_threshold=10,
                         re_init_dt=re_init_dt,
                         dist_param=5.5,
                         scale=1,
                         distribution='normal',
                         clip_min=4,
                         clip_max=7,
                         unit='ms',
                         reference='synapse_counter')

##################
# Setting up monitors
spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
spikemon_seq_neurons = SpikeMonitor(poisson_spikes, name='spikemon_seq_neurons')
statemon_wplast = StateMonitor(feedforward_exc, variables=['w_plast'],
                               record=True, name='statemon_wplast')
statemon_weight = StateMonitor(feedforward_exc, variables=['weight'],
                               record=True, name='statemon_weight')
statemon_counter = StateMonitor(feedforward_exc, variables=['re_init_counter'],
                               record=True, name='statemon_counter')
cell_mon = StateMonitor(exc_cells, variables=['Vm'], record=True, name='exc')

net = TeiliNetwork()
net.add(exc_cells, poisson_spikes, feedforward_exc, spikemon_exc_neurons,
        spikemon_seq_neurons, statemon_wplast, statemon_weight,
        statemon_counter, cell_mon)
net.run(sim_time*ms, report='stdout', report_period=100*ms)

# Plots
from teili.tools.visualizer.DataModels import EventsModel, StateVariablesModel
from teili.tools.visualizer.DataControllers import Rasterplot, Lineplot
from teili.tools.visualizer.DataViewers import PlotSettings
import pyqtgraph as pg
from PyQt5 import QtGui

QtApp = QtGui.QApplication([])

colors = [
    (0, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 255, 255)
]
cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 4), color=colors)

win1 = QtGui.QMainWindow()
image_axis = pg.PlotItem()
image_axis.setLabel(axis='bottom', text='Neuron index')
image_axis.setLabel(axis='left', text='Input channels')
m1 = pg.ImageView(view=image_axis)
win1.setCentralWidget(m1)
win1.show()
m1.setImage(np.reshape(statemon_wplast.w_plast, (num_inputs, num_exc, -1)), axes={'t':2, 'y':0, 'x':1})
m1.setColorMap(cmap)
image_axis = pg.PlotItem()
image_axis.setLabel(axis='bottom', text='postsynaptic neuron')
image_axis.setLabel(axis='left', text='presynaptic neuron')

win2 = QtGui.QMainWindow()
image_axis = pg.PlotItem()
image_axis.setLabel(axis='bottom', text='Neuron index')
image_axis.setLabel(axis='left', text='Input channels')
m2 = pg.ImageView(view=image_axis)
win2.setCentralWidget(m2)
win2.show()
m2.setImage(np.reshape(statemon_weight.weight, (num_inputs, num_exc, -1)), axes={'t':2, 'y':0, 'x':1})
m2.setColorMap(cmap)
image_axis = pg.PlotItem()
image_axis.setLabel(axis='bottom', text='postsynaptic neuron')
image_axis.setLabel(axis='left', text='presynaptic neuron')

exc_raster = EventsModel.from_brian_spike_monitor(spikemon_exc_neurons)
seq_raster = EventsModel.from_brian_spike_monitor(spikemon_seq_neurons)

state_variable_names = ['re_init_counter']
counter_copy = np.array(statemon_counter.re_init_counter)
counter_copy[np.isnan(counter_copy)] = 0
state_variables = [counter_copy]
state_variables_times = [statemon_counter.t/ms]
counter_line = StateVariablesModel(state_variable_names, state_variables, state_variables_times)

#line_plot1 = Lineplot(DataModel_to_x_and_y_attr=[(counter_line, ('t_re_init_counter', 're_init_counter'))],
#                      title='reinit counters with time', xlabel='time (s)',
#                      ylabel='counter value', backend='pyqtgraph', QtApp=QtApp)
raster_plot1 = Rasterplot(MyEventsModels=[exc_raster], backend='pyqtgraph', QtApp=QtApp)
raster_plot2 = Rasterplot(MyEventsModels=[seq_raster], backend='pyqtgraph', QtApp=QtApp,
                show_immediately=True)
from brian2 import *
plot(counter_line.re_init_counter.T)
show()
