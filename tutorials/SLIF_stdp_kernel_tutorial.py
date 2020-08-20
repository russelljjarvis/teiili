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
from teili.models.synapse_models import StochasticSyn_decay_stoch_stdp
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


stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')
stdp_synapse = Connections(pre_neurons, post_neurons,
                method=stochastic_decay,
                equation_builder=StochasticSyn_decay_stoch_stdp(),
                name='stdp_synapse')

stdp_synapse.connect('i==j')

# Setting parameters
stdp_synapse.w_plast = 7
stdp_synapse.dApre = 7
stdp_synapse.taupre = 10*ms
stdp_synapse.taupost = 10*ms
add_lfsr(stdp_synapse, 12, defaultclock.dt)


spikemon_pre_neurons = SpikeMonitor(pre_neurons, record=True)
spikemon_post_neurons = SpikeMonitor(post_neurons, record=True)
statemon_post_synapse = StateMonitor(stdp_synapse, variables=[
    'decay_probability_stdp', 'Apre', 'Apost'],
    record=(48,47,46), name='statemon_post_synapse')

run(tmax + 1 * ms)


if visualization_backend == 'pyqtgraph':
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)
    else:
        print('QApplication instance already exists: %s' % str(app))
else:
    app=None

win_1 = pg.GraphicsWindow(title="1")
datamodel = StateVariablesModel(state_variable_names=['w_plast'],
                                state_variables=[stdp_synapse.w_plast],
                                state_variables_times=[np.asarray((post_neurons.tspike - pre_neurons.tspike) / ms)])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_w_plast', 'w_plast'))],
        title="Spike-time dependent plasticity",
        xlabel='\u0394 t',  # delta t
        ylabel='w',
        backend=visualization_backend,
        QtApp=app,
        mainfig=win_1,
        show_immediately=False)

win_2 = pg.GraphicsWindow(title="2")
Lineplot(DataModel_to_x_and_y_attr=[(statemon_post_synapse[46], ('t', 'Apre')), (statemon_post_synapse[46], ('t', 'Apost'))],
        title="Apre",
        xlabel='time',  # delta t
        ylabel='Apre',
        backend=visualization_backend,
        QtApp=app,
        mainfig=win_2,
        show_immediately=False)

win_3 = pg.GraphicsWindow(title="3")
Lineplot(DataModel_to_x_and_y_attr=[(statemon_post_synapse[47], ('t', 'Apre')), (statemon_post_synapse[47], ('t', 'Apost'))],
        title="Apost",
        xlabel='time',  # delta t
        ylabel='Apost',
        backend=visualization_backend,
        QtApp=app,
        mainfig=win_3,
        show_immediately=False)

win_4 = pg.GraphicsWindow(title="4")
Lineplot(DataModel_to_x_and_y_attr=[(statemon_post_synapse[48], ('t', 'Apre')), (statemon_post_synapse[48], ('t', 'Apost'))],
        title="Apost",
        xlabel='time',  # delta t
        ylabel='Apost',
        backend=visualization_backend,
        QtApp=app,
        mainfig=win_4,
        show_immediately=False)

win_5 = pg.GraphicsWindow(title="5")
Lineplot(DataModel_to_x_and_y_attr=[(statemon_post_synapse, ('t', 'decay_probability_stdp'))],
        title="decay_probability_stdp",
        xlabel='time',  # delta t
        ylabel='decay_probability_stdp',
        backend=visualization_backend,
        QtApp=app,
        mainfig=win_5,
        show_immediately=False)

win_6 = pg.GraphicsWindow(title="6")
Rasterplot(MyEventsModels=[spikemon_pre_neurons, spikemon_post_neurons],
            MyPlotSettings=PlotSettings(colors=['w', 'r']),
            title='',
            xlabel='Time (s)',
            ylabel='Neuron ID',
            backend=visualization_backend,
            QtApp=app,
            mainfig=win_6,
            show_immediately=True)
