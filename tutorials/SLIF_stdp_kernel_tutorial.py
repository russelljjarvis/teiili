"""
Created on 30.11.2017

@author: Moritz Milde
Email: mmilde@ini.uzh.ch

This script is adapted from https://code.ini.uzh.ch/alpren/gridcells/blob/master/STDP_IE_HaasKernel.py

This script contains a simple event based way to simulate complex STDP kernels
"""

from brian2 import ms, prefs, StateMonitor, SpikeMonitor, run, defaultclock,\
        ExplicitStateUpdater
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import numpy as np

from teili.core.groups import Neurons, Connections
from teili.models.synapse_models import DPIstdp, StochasticSyn_decay_stdp
from teili.tools.add_run_reg import add_lfsr

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import StateVariablesModel
from teili.tools.visualizer.DataControllers import Lineplot, Rasterplot

prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms
visualization_backend = 'pyqtgraph'  # Or set it to 'matplotlib' to use matplotlib.pyplot to plot


font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }


tmax = 99 * ms
N = 100

# Presynaptic neurons G spike at times from 0 to tmax
# Postsynaptic neurons G spike at times from tmax to 0
# So difference in spike times will vary from -tmax to +tmax
pre_neurons = Neurons(N, model='''tspike:second''', threshold='t>tspike', refractory=100 * ms)

pre_neurons.namespace.update({'tmax': tmax})
post_neurons = Neurons(N, model='''
                Iin0 : amp
                tspike:second''', threshold='t>tspike', refractory=100 * ms)
post_neurons.namespace.update({'tmax': tmax})

pre_neurons.tspike = 'i*tmax/(N-1)'
post_neurons.tspike = '(N-1-i)*tmax/(N-1)'


stochastic_decay = ExplicitStateUpdater('''x_new = dt*f(x,t)''')
stdp_synapse = Connections(pre_neurons, post_neurons,
                method=stochastic_decay,
                equation_builder=StochasticSyn_decay_stdp(),
                name='stdp_synapse')

stdp_synapse.connect('i==j')

# Setting parameters
stdp_synapse.w_plast = 7
stdp_synapse.w_max = 15
stdp_synapse.dApre = 4
stdp_synapse.taupre = 30 * ms
stdp_synapse.taupost = 30 * ms
stdp_synapse.weight = 1
add_lfsr(stdp_synapse, 12, defaultclock.dt)


spikemon_pre_neurons = SpikeMonitor(pre_neurons, record=True)
spikemon_post_neurons = SpikeMonitor(post_neurons, record=True)
statemon_post_synapse = StateMonitor(stdp_synapse, variables=[
    'decay_probability_stdp', 'dApre'],
    record=True, name='statemon_post_synapse')

run(tmax + 1 * ms)


if visualization_backend == 'pyqtgraph':
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)
    else:
        print('QApplication instance already exists: %s' % str(app))
else:
    app=None

datamodel = StateVariablesModel(state_variable_names=['w_plast'],
                                state_variables=[stdp_synapse.w_plast],
                                state_variables_times=[np.asarray((post_neurons.tspike - pre_neurons.tspike) / ms)])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_w_plast', 'w_plast'))],
        title="Spike-time dependent plasticity",
        xlabel='\u0394 t',  # delta t
        ylabel='w',
        backend=visualization_backend,
        QtApp=app,
        show_immediately=False)

#Lineplot(DataModel_to_x_and_y_attr=[(statemon_post_synapse, ('t', 'dApre'))],
#        title="dApre",
#        xlabel='time',  # delta t
#        ylabel='dApre',
#        backend=visualization_backend,
#        QtApp=app,
#        show_immediately=False)

Rasterplot(MyEventsModels=[spikemon_pre_neurons, spikemon_post_neurons],
            MyPlotSettings=PlotSettings(colors=['w', 'r']),
            title='',
            xlabel='Time (s)',
            ylabel='Neuron ID',
            backend=visualization_backend,
            QtApp=app,
            show_immediately=True)
