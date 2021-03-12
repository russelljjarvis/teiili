import numpy as np

from brian2 import Hz, ms, mV, prefs, SpikeMonitor, StateMonitor,\
        defaultclock, PoissonGroup

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

from teili.tools.visualizer.DataModels import EventsModel, StateVariablesModel
from teili.tools.visualizer.DataControllers import Rasterplot, Lineplot
import pyqtgraph as pg
from PyQt5 import QtGui

#############
# Load models
syn_model = SynapseEquationBuilder(base_unit='current',
                                   kernel='exponential',
                                   plasticity='non_plastic')
adapt_neu_model = NeuronEquationBuilder(base_unit='voltage',
                                        leak='leaky',
                                        position='spatial',
                                        intrinsic_excitability='threshold_adaptation')

# Initialize simulation preferences
prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms

# Initialize input sequence: Poisson rates shaped like a gaussian
num_inputs = 20
input_base_rate = 5*Hz
input_space = np.array([x for x in range(num_inputs)])
rate_distribution = 500 * np.exp(-(input_space - 10)**2 / (2 * (1)**2)) * Hz

poisson_spikes = PoissonGroup(num_inputs, rate_distribution + input_base_rate)

#################
# Building network
num_exc = 20
exc_cells = Neurons(num_exc,
                    equation_builder=adapt_neu_model(num_inputs=3),
                    name='exc_cells',
                    verbose=True)

feedforward_exc = Connections(poisson_spikes, exc_cells,
                              equation_builder=syn_model(),
                              name='feedforward_exc')
feedforward_exc.connect(j='i')

# Parameters
feedforward_exc.tausyn = 5*ms
exc_cells.Vm = exc_cells.EL
feedforward_exc.weight = 0.1
exc_cells.thr_min = -69*mV
exc_cells.thr_max = -40*mV

##################
# Setting up monitors
spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
spikemon_poisson = SpikeMonitor(poisson_spikes,
                                name='spikemon_poisson')
statemon_thresh = StateMonitor(exc_cells, variables=['Vthr'],
                               record=True,
                               name='statemon_thresh')

net = TeiliNetwork()
net.add(poisson_spikes, exc_cells, feedforward_exc, spikemon_exc_neurons,
        spikemon_poisson, statemon_thresh)
net.run(50000*ms, report='stdout', report_period=100*ms)

# Plots
QtApp = QtGui.QApplication([])

exc_raster = EventsModel.from_brian_spike_monitor(spikemon_exc_neurons)
in_raster = EventsModel.from_brian_spike_monitor(spikemon_poisson)
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
RC = Rasterplot(MyEventsModels=[in_raster],
                title='Input spikes',
                ylabel='Indices',
                xlabel='Time (s)',
                backend='pyqtgraph',
                QtApp=QtApp,
                mainfig=win2)
win3 = pg.GraphicsWindow()
LC = Lineplot(DataModel_to_x_and_y_attr=[(thresh_traces, ('t_Vthr', 'Vthr'))],
              title='Threshold decay of all neurons',
              xlabel='time (s)',
              ylabel='Vth (V)',
              backend='pyqtgraph',
              QtApp=QtApp,
              mainfig=win3,
              show_immediately=True)
