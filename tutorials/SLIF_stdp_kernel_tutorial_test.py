"""
Created on 30.11.2017

@author: Moritz Milde
Email: mmilde@ini.uzh.ch

This script is adapted from https://code.ini.uzh.ch/alpren/gridcells/blob/master/STDP_IE_HaasKernel.py

This script contains a simple event based way to simulate complex STDP kernels
"""

from brian2 import ms, prefs, StateMonitor, SpikeMonitor, run, defaultclock,\
        ExplicitStateUpdater, TimedArray
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import numpy as np
import os

from teili.core.groups import Neurons, Connections
#from teili.models.synapse_models import StochasticSyn_decay_stoch_stdp
from teili.tools.add_run_reg import add_lfsr
from lfsr import create_lfsr
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import StateVariablesModel
from teili.tools.visualizer.DataControllers import Lineplot, Rasterplot

prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms
visualization_backend = 'pyqtgraph'  # Or set it to 'matplotlib' to use matplotlib.pyplot to plot

path = os.path.expanduser("/home/pablo/git/teili")
model_path = os.path.join(path, "teili", "models", "equations", "")
StochasticSyn_decay_stoch_stdp = SynapseEquationBuilder.import_eq(
        model_path + 'StochStdpNew.py')

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }


wait_time = 0*ms#TODO
trials = 100
trial_duration = 50
tmax = trial_duration*trials
N = 50

# Presynaptic neurons G spike at times from 0 to tmax
# Postsynaptic neurons G spike at times from tmax to 0
# So difference in spike times will vary from -tmax to +tmax
pre_neurons = Neurons(N, model='v = ta_pre(t, i) : 1',
                      threshold='v == 1', refractory='1*ms')
pre_neurons.namespace.update({'tmax': tmax})

post_neurons = Neurons(N, model='''v = ta_post(t, i) : 1
                                   Iin0 : amp''',
                       threshold='v == 1', refractory='1*ms')
post_neurons.namespace.update({'tmax': tmax})

post_tspikes = np.arange(1, N*trials + 1).reshape((trials, N))
pre_tspikes = post_tspikes[:, np.array(range(N-1, -1, -1))]
pre_input = np.zeros((tmax, N))
post_input = np.zeros((tmax, N))
for ind, spks in enumerate(pre_tspikes.T):
    pre_input[spks.astype(int)-1, ind] = 1
for ind, spks in enumerate(post_tspikes.T):
    post_input[spks.astype(int)-1, ind] = 1
ta_pre = TimedArray(pre_input, dt=defaultclock.dt)
ta_post = TimedArray(post_input, dt=defaultclock.dt)
pre_neurons.namespace.update({'ta_pre':ta_pre})
post_neurons.namespace.update({'ta_post':ta_post})

stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')
stdp_synapse = Connections(pre_neurons, post_neurons,
                method=stochastic_decay,
                equation_builder=StochasticSyn_decay_stoch_stdp(),
                name='stdp_synapse')

stdp_synapse.connect('i==j')

# Setting parameters
stdp_synapse.w_plast = 7
stdp_synapse.taupre = 10*ms
stdp_synapse.taupost = 10*ms
stdp_synapse.stdp_thres = 2
ta = create_lfsr([], [stdp_synapse], defaultclock.dt)


spikemon_pre_neurons = SpikeMonitor(pre_neurons, record=True)
spikemon_post_neurons = SpikeMonitor(post_neurons, record=True)
statemon_post_synapse = StateMonitor(stdp_synapse, variables=[
    'Apre', 'Apost'],
    record=(48,47,46), name='statemon_post_synapse')
statemon_pre_neurons = StateMonitor(pre_neurons, variables=['v'],
    record=True, name='statemon_pre_neurons')
statemon_post_neurons = StateMonitor(post_neurons, variables=['v'],
    record=True, name='statemon_post_neurons')

run(tmax*ms + 1*ms)


if visualization_backend == 'pyqtgraph':
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)
    else:
        print('QApplication instance already exists: %s' % str(app))
else:
    app=None

# TODO get w_plast of a given neurons (which corresponds to a single delta t), take average and plot
#win_1 = pg.GraphicsWindow(title="1")
#datamodel = StateVariablesModel(state_variable_names=['w_plast'],
#                                state_variables=[stdp_synapse.w_plast],
#                                state_variables_times=[np.asarray((post_neurons.tspike - pre_neurons.tspike) / ms)])
#Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_w_plast', 'w_plast'))],
#        title="Spike-time dependent plasticity",
#        xlabel='\u0394 t',  # delta t
#        ylabel='w',
#        backend=visualization_backend,
#        QtApp=app,
#        mainfig=win_1,
#        show_immediately=False)
#
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

win_5 = pg.GraphicsWindow(title="6")
Rasterplot(MyEventsModels=[spikemon_pre_neurons, spikemon_post_neurons],
            MyPlotSettings=PlotSettings(colors=['w', 'r']),
            title='',
            xlabel='Time (s)',
            ylabel='Neuron ID',
            backend=visualization_backend,
            QtApp=app,
            mainfig=win_5,
            show_immediately=True)
