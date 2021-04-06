import numpy as np
import sys

from brian2 import Hz, mA, ms, mV, prefs, SpikeMonitor, StateMonitor,\
        defaultclock, PoissonGroup

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.builder.synapse_equation_builder import\
        SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

from teili.tools.visualizer.DataModels import EventsModel, StateVariablesModel
from teili.tools.visualizer.DataControllers import Rasterplot, Lineplot
from teili.tools.lfsr import create_lfsr
import pyqtgraph as pg
from PyQt5 import QtGui

# Load models
syn_model = SynapseEquationBuilder(base_unit='current',
                                   kernel='exponential',
                                   plasticity='non_plastic')
adapt_neu_model = NeuronEquationBuilder(base_unit='voltage',
                                        leak='leaky',
                                        position='spatial',
                                        intrinsic_excitability='threshold_adaptation')
""" Alternatively, use the stochastic quantized model
syn_model = SynapseEquationBuilder(base_unit='quantized',
                                   #rand_gen='lfsr_syn',
                                   plasticity='non_plastic')
adapt_neu_model = NeuronEquationBuilder(base_unit='quantized',
                                        #rand_gen='lfsr',
                                        intrinsic_excitability='threshold_adaptation',
                                        position='spatial')
"""

# Initialize simulation preferences
prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms

# Initialize input sequence: Poisson rates shaped like a gaussian
num_inputs = 20
input_base_rate = 10*Hz
input_space = np.array([x for x in range(num_inputs)])
rate_distribution = 50 * np.exp(-(input_space - 10)**2 / (2 * (1)**2)) * Hz

poisson_spikes = PoissonGroup(num_inputs, rate_distribution + input_base_rate)

# Building network
if 'decay_probability' in adapt_neu_model.keywords['model']:
    from brian2 import ExplicitStateUpdater
    method = ExplicitStateUpdater('''x_new = f(x,t)''')
elif 'Vm' in adapt_neu_model.keywords['model']:
    method = 'euler'
num_exc = num_inputs
exc_cells = Neurons(num_exc,
                    equation_builder=adapt_neu_model(num_inputs=3),
                    name='exc_cells',
                    method=method,
                    verbose=True)

feedforward_exc = Connections(poisson_spikes, exc_cells,
                              equation_builder=syn_model(),
                              method=method,
                              name='feedforward_exc')
feedforward_exc.connect(j='i')

# Parameters
feedforward_exc.tausyn = 8*ms
exc_cells.tau_thr = 30000*ms
if 'decay_probability' in adapt_neu_model.keywords['model']:
    feedforward_exc.gain_syn = 15*mA
    exc_cells.Vm = exc_cells.Vrest
    exc_cells.tau = 5*ms
    exc_cells.thr_min = 4*mV
    exc_cells.thr_max = 15*mV
    if ('lfsr' in adapt_neu_model.keywords['model']
            and 'lfsr' in syn_model.keywords['model']):
        feedforward_exc.lfsr_num_bits_syn = 5
        feedforward_exc.lfsr_num_bits_Apre = 5
        feedforward_exc.lfsr_num_bits_Apost = 5
        exc_cells.lfsr_num_bits = 5
        ta = create_lfsr([exc_cells],
                         [feedforward_exc],
                         defaultclock.dt)
elif 'Vm' in adapt_neu_model.keywords['model']:
    exc_cells.Vm = exc_cells.EL
    feedforward_exc.weight = 1
    exc_cells.thr_min = -69*mV
    exc_cells.thr_max = -40*mV

##################
# Setting up monitors
spikemon_exc_neurons = SpikeMonitor(exc_cells, name='spikemon_exc_neurons')
spikemon_poisson = SpikeMonitor(poisson_spikes,
                                name='spikemon_poisson')
statemon_thresh = StateMonitor(exc_cells, variables=['Vthr', 'Vm'],
                               record=True,
                               name='statemon_thresh')

net = TeiliNetwork()
net.add(poisson_spikes, exc_cells, feedforward_exc, spikemon_exc_neurons,
        spikemon_poisson, statemon_thresh)
net.run(40000*ms, report='stdout', report_period=100*ms)

# Plots
app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))
QtApp = QtGui.QApplication([])

win = pg.GraphicsWindow()
win.resize(2100, 1200)
win.setWindowTitle('Threshold adaptation')

p1 = win.addPlot(title='Neuron spikes')
win.nextRow()
p2 = win.addPlot(title='Input raster')
win.nextRow()
p3 = win.addPlot(title='Threshold')

exc_raster = EventsModel.from_brian_spike_monitor(spikemon_exc_neurons)
in_raster = EventsModel.from_brian_spike_monitor(spikemon_poisson)
skip_not_rec_neuron_ids = True
thresh_traces = StateVariablesModel.from_brian_state_monitors(
        [statemon_thresh], skip_not_rec_neuron_ids)

RC = Rasterplot(MyEventsModels=[in_raster],
                title='Input spikes',
                ylabel='Indices',
                xlabel='Time (s)',
                backend='pyqtgraph',
                QtApp=QtApp,
                subfig_rasterplot=p1,
                mainfig=win)
RC = Rasterplot(MyEventsModels=[exc_raster],
                backend='pyqtgraph',
                title='Spikes from excitatory neurons',
                ylabel='Indices',
                xlabel='Time (s)',
                QtApp=QtApp,
                subfig_rasterplot=p2,
                mainfig=win)
LC = Lineplot(DataModel_to_x_and_y_attr=[(thresh_traces, ('t_Vthr', 'Vthr'))],
              title='Threshold decay of all neurons',
              xlabel='Time (s)',
              ylabel='Vth (V)',
              backend='pyqtgraph',
              QtApp=QtApp,
              subfig=p3,
              mainfig=win,
              show_immediately=True)
