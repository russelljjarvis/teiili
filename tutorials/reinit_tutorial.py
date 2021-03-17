import numpy as np

from brian2 import Hz, second, ms, mV, prefs, SpikeMonitor, StateMonitor,\
        defaultclock, PoissonGroup

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.stimuli.testbench import SequenceTestbench
from teili.tools.group_tools import add_group_params_re_init,\
        add_group_param_init
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

from teili.tools.visualizer.DataModels import EventsModel, StateVariablesModel
from teili.tools.visualizer.DataControllers import Rasterplot, Lineplot
from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.lfsr import create_lfsr
from teili.tools.misc import neuron_group_from_spikes
import pyqtgraph as pg
from PyQt5 import QtGui

#############
# Load models
#syn_model=SynapseEquationBuilder(base_unit='current',
#                                 kernel='exponential',
#                                 plasticity='stdp',
#                                 structural_plasticity='deterministic_counter')
#neu_model=NeuronEquationBuilder(base_unit='voltage',
#                                leak='leaky',
#                                position='spatial')
# Alternatively, use the stochastic quantized model
syn_model=SynapseEquationBuilder(base_unit='QuantizedStochastic',
                                 plasticity='quantized_stochastic_stdp',
                                 #rand_gen='lfsr_syn',
                                 structural_plasticity='stochastic_counter')
neu_model=NeuronEquationBuilder(base_unit='quantized',
                                #rand_gen='lfsr',
                                position='spatial')

# Initialize simulation preferences
prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms
sim_duration = 10000 * defaultclock.dt
total_sim_duration = 2*sim_duration

# Initialize input sequence: Poisson rates shaped like a gaussian
num_inputs = 20
input_base_rate = 0*Hz
input_space = np.array([x for x in range(num_inputs)])
rate_distribution = 500 * np.exp(-(input_space - 5)**2 / (2 * (1)**2)) * Hz
poisson_spikes = PoissonGroup(num_inputs, rate_distribution + input_base_rate)

net = TeiliNetwork()
temp_monitor = SpikeMonitor(poisson_spikes, name='temp_monitor')
net.add(poisson_spikes, temp_monitor)
net.run(sim_duration)

# Change pattern
rate_distribution = 500 * np.exp(-(input_space - 15)**2 / (2 * (1)**2)) * Hz
poisson_spikes.rates = rate_distribution + input_base_rate
net.run(sim_duration)
temp_monitor.active = False

# Convert Poisson group activity to a neuron group activity. This is required
# for quantized stochastic models
poisson_spikes = neuron_group_from_spikes(num_inputs, defaultclock.dt,
                                          total_sim_duration,
                                          spike_indices=np.array(temp_monitor.i),
                                          spike_times=np.array(temp_monitor.t)*second)

#################
# Building network
# cells
num_exc = num_inputs
if 'decay_probability' in neu_model.keywords['model']:
    from brian2 import ExplicitStateUpdater
    method = ExplicitStateUpdater('''x_new = f(x,t)''')
elif 'Vm' in neu_model.keywords['model']:
    method = 'euler'
exc_cells = Neurons(num_exc,
                    equation_builder=neu_model(num_inputs=1),
                    name='exc_cells',
                    method=method,
                    verbose=True)

# Connections
feedforward_exc = Connections(poisson_spikes,
                              exc_cells,
                              equation_builder=syn_model(),
                              method=method,
                              name='feedforward_exc')
#feedforward_exc.connect(j='i')
feedforward_exc.connect()

feedforward_exc.weight = 1
num_connections = len(feedforward_exc.weight)
sparsity = .9
ffe_zero_w = np.random.choice(num_connections,
                              int(num_connections*sparsity),
                              replace=False)
feedforward_exc.weight[ffe_zero_w] = 0
feedforward_exc.w_plast[ffe_zero_w] = 0

# Initializations
feedforward_exc.tausyn = 5*ms
if 'decay_probability' in neu_model.keywords['model']:
    variable_type = 'int'
    wplast_min = 0
    wplast_max = 15
    wplast_scale = 1
    exc_cells.tau = 20*ms
    exc_cells.Vm = exc_cells.Vrest
    feedforward_exc.rand_num_bits_Apre = 4
    feedforward_exc.rand_num_bits_Apost = 4
    feedforward_exc.taupre = 20*ms
    feedforward_exc.taupost = 20*ms
    # LFSR
    if 'lfsr' in neu_model.keywords['model'] and 'lfsr' in syn_model.keywords['model']:
        feedforward_exc.lfsr_num_bits_syn = 3
        feedforward_exc.lfsr_num_bits_Apre = 5
        feedforward_exc.lfsr_num_bits_Apost = 5
        exc_cells.lfsr_num_bits = 5
        ta = create_lfsr([exc_cells],
                         [feedforward_exc],
                         defaultclock.dt)
elif 'Vm' in neu_model.keywords['model']:
    variable_type = 'float'
    wplast_min = 0
    wplast_max = 1
    wplast_scale = .05
    exc_cells.Vm = exc_cells.EL
    feedforward_exc.dApre = 0.001

add_group_param_init(groups=[feedforward_exc],
                     variable='w_plast',
                     dist_param=1,
                     scale=.05,
                     distribution='gamma',
                     clip_min=wplast_min,
                     clip_max=wplast_max)

##################
# Synaptic homeostasis
re_init_dt = 1000*ms
add_group_params_re_init(groups=[feedforward_exc],
                         variable='w_plast',
                         re_init_variable='re_init_counter',
                         re_init_threshold=1,
                         re_init_dt=re_init_dt,
                         dist_param=1,
                         scale=wplast_scale,
                         distribution='gamma',
                         clip_min=wplast_min,
                         clip_max=wplast_max,
                         variable_type=variable_type,
                         reference='synapse_counter')
add_group_params_re_init(groups=[feedforward_exc],
                         variable='weight',
                         re_init_variable='re_init_counter',
                         re_init_threshold=1,
                         re_init_dt=re_init_dt,
                         distribution='deterministic',
                         const_value=1,
                         reference='synapse_counter')
add_group_params_re_init(groups=[feedforward_exc],
                         variable='tausyn',
                         re_init_variable='re_init_counter',
                         re_init_threshold=1,
                         re_init_dt=re_init_dt,
                         dist_param=5.5,
                         scale=1,
                         distribution='normal',
                         clip_min=4,
                         clip_max=7,
                         variable_type=variable_type,
                         unit='ms',
                         reference='synapse_counter')

##################
# Setting up monitors
spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
spikemon_seq_neurons = SpikeMonitor(poisson_spikes, name='spikemon_seq_neurons')
statemon_ffe = StateMonitor(feedforward_exc,
                            variables=['w_plast', 'weight', 're_init_counter', 'Apre'],
                            record=True, name='statemon_ffe')
cell_mon = StateMonitor(exc_cells, variables=['Vm'], record=True, name='exc')

net = TeiliNetwork()
net.add(exc_cells, poisson_spikes, feedforward_exc, spikemon_exc_neurons,
        spikemon_seq_neurons, statemon_ffe, cell_mon)
net.run(total_sim_duration, report='stdout', report_period=100*ms)

# Plots
app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))
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
m1.setImage(np.reshape(statemon_ffe.w_plast*(statemon_ffe.weight>0),
                       (int(num_connections/num_inputs),
                       num_exc,
                       -1)),
            axes={'t':2, 'y':0, 'x':1})
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
m2.setImage(np.reshape(statemon_ffe.weight,
                       (int(num_connections/num_inputs), num_exc, -1)),
            axes={'t':2, 'y':0, 'x':1})
m2.setColorMap(cmap)
image_axis = pg.PlotItem()
image_axis.setLabel(axis='bottom', text='postsynaptic neuron')
image_axis.setLabel(axis='left', text='presynaptic neuron')

exc_raster = EventsModel.from_brian_spike_monitor(spikemon_exc_neurons)
seq_raster = EventsModel.from_brian_spike_monitor(spikemon_seq_neurons)

state_variable_names = ['re_init_counter']
counter_copy = np.array(statemon_ffe.re_init_counter)
counter_copy[np.isnan(counter_copy)] = 0
state_variables = [counter_copy.T]
state_variables_times = [statemon_ffe.t/ms]
counter_line = StateVariablesModel(state_variable_names, state_variables, state_variables_times)

line_plot1 = Lineplot(DataModel_to_x_and_y_attr=[(counter_line, ('t_re_init_counter', 're_init_counter'))],
                      title='Reinit counters with time',
                      xlabel='time (s)',
                      ylabel='counter value',
                      backend='pyqtgraph',
                      QtApp=QtApp)
raster_plot1 = Rasterplot(MyEventsModels=[exc_raster],
                          title='Raster plot of neurons',
                          xlabel='time (s)',
                          ylabel='Indices',
                          backend='pyqtgraph',
                          QtApp=QtApp)
raster_plot2 = Rasterplot(MyEventsModels=[seq_raster],
                          title='Raster plot of input',
                          xlabel='time (s)',
                          ylabel='Indices',
                          backend='pyqtgraph',
                          QtApp=QtApp,
                          show_immediately=True)
