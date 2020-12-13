"""
This script contains a simple event based way to simulate a stochastic
STDP kernel.
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
visualization_backend = 'pyqtgraph'

path = os.path.expanduser("/home/pablo/git/teili")
model_path = os.path.join(path, "teili", "models", "equations", "")
StochasticSyn_decay_stoch_stdp = SynapseEquationBuilder.import_eq(
        model_path + 'StochStdpNew.py')

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }


trials = 105
trial_duration = 50
wait_time = 100  # set delay between trials to avoid interferences
tmax = trial_duration*trials + wait_time*trials
N = 50

# Define matched spike times between pre and post neurons
post_tspikes = np.arange(1, N*trials + 1).reshape((trials, N))
pre_tspikes = post_tspikes[:, np.array(range(N-1, -1, -1))]

# Create inputs arrays, which will be 1 when neurons are supposed to spike
pre_input = np.zeros((tmax, N))
post_input = np.zeros((tmax, N))
for ind, spks in enumerate(pre_tspikes.T):
    for j, spk in enumerate(spks.astype(int)):
        pre_input[spk-1 + j*wait_time, ind] = 1
for ind, spks in enumerate(post_tspikes.T):
    for j, spk in enumerate(spks.astype(int)):
        post_input[spk-1 + j*wait_time, ind] = 1

ta_pre = TimedArray(pre_input, dt=defaultclock.dt)
ta_post = TimedArray(post_input, dt=defaultclock.dt)

average_trials = 100
average_wplast = np.zeros((average_trials, 50))
for avg_trial in range(average_trials):
    pre_neurons = Neurons(N, model='v = ta_pre(t, i) : 1',
                          threshold='v == 1', refractory='1*ms')
    pre_neurons.namespace.update({'tmax': tmax})
    pre_neurons.namespace.update({'ta_pre': ta_pre})

    post_neurons = Neurons(N, model='''v = ta_post(t, i) : 1
                                       Iin0 : amp''',
                           threshold='v == 1', refractory='1*ms')
    post_neurons.namespace.update({'tmax': tmax})
    post_neurons.namespace.update({'ta_post': ta_post})

    stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')
    stdp_synapse = Connections(pre_neurons, post_neurons,
                               method=stochastic_decay,
                               equation_builder=StochasticSyn_decay_stoch_stdp(),
                               name='stdp_synapse')
    stdp_synapse.connect('i==j')

    # Setting parameters
    stdp_synapse.w_plast = 7
    stdp_synapse.taupre = 20*ms
    stdp_synapse.taupost = 20*ms
    stdp_synapse.stdp_thres = 1
    stdp_synapse.lfsr_num_bits_Apre = 5
    stdp_synapse.lfsr_num_bits_Apost = 5

    ta = create_lfsr([], [stdp_synapse], defaultclock.dt)
    # Avoids alignment between LFSR numbers. Effectively, it 
    # skips last value of the LFSR (considering it is 4 bits long)
    stdp_synapse.lfsr_max_value_condApost2 = 14*ms
    stdp_synapse.lfsr_max_value_condApre2 = 14*ms

    spikemon_pre_neurons = SpikeMonitor(pre_neurons, record=True)
    spikemon_post_neurons = SpikeMonitor(post_neurons, record=True)
    statemon_synapse = StateMonitor(stdp_synapse,
                                    variables=['Apre', 'Apost', 'w_plast',
                                               'cond_Apost1', 'cond_Apost2',
                                               'Apre1_lfsr', 'Apre2_lfsr',
                                               'Apost1_lfsr', 'Apost2_lfsr',
                                               'decay_probability_Apre',
                                               'decay_probability_Apost'],
                                    record=True,
                                    name='statemon_synapse')

    run(tmax*ms)
    average_wplast[avg_trial, :] = np.array(stdp_synapse.w_plast)

    if visualization_backend == 'pyqtgraph':
        app = QtGui.QApplication.instance()
        if app is None:
            app = QtGui.QApplication(sys.argv)
        else:
            print('QApplication instance already exists: %s' % str(app))
    else:
        app = None

pairs_timing = (spikemon_post_neurons.t[:trial_duration]
                - spikemon_post_neurons.t[:trial_duration][::-1])/ms
win_1 = pg.GraphicsWindow(title="1")
datamodel = StateVariablesModel(state_variable_names=['w_plast'],
                                state_variables=[average_wplast[avg_trial, :]],
                                state_variables_times=[pairs_timing])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_w_plast', 'w_plast'))],
        title="Spike-time dependent plasticity (trial)",
        xlabel='\u0394 t (ms)',  # delta t
        ylabel='w',
        backend=visualization_backend,
        QtApp=app,
        mainfig=win_1,
        show_immediately=False)

win_2 = pg.GraphicsWindow(title="1")
datamodel = StateVariablesModel(state_variable_names=['w_plast'],
                                state_variables=[average_wplast[avg_trial-1, :]],
                                state_variables_times=[pairs_timing])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_w_plast', 'w_plast'))],
        title="Spike-time dependent plasticity (trial)",
        xlabel='\u0394 t (ms)',  # delta t
        ylabel='w',
        backend=visualization_backend,
        QtApp=app,
        mainfig=win_2,
        show_immediately=False)

win_3 = pg.GraphicsWindow(title="1")
datamodel = StateVariablesModel(state_variable_names=['w_plast'],
                                state_variables=[np.mean(average_wplast, axis=0)],
                                state_variables_times=[pairs_timing])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_w_plast', 'w_plast'))],
        title="Spike-time dependent plasticity (average)",
        xlabel='\u0394 t (ms)',  # delta t
        ylabel='w',
        backend=visualization_backend,
        QtApp=app,
        mainfig=win_3,
        show_immediately=False)
    #win_2 = pg.GraphicsWindow(title="2")
    #Lineplot(DataModel_to_x_and_y_attr=[(statemon_synapse[8], ('t', 'Apre')), (statemon_synapse[0], ('t', 'Apost'))],
    #        title="Apre",
    #        xlabel='time',  # delta t
    #        ylabel='Apre',
    #        backend=visualization_backend,
    #        QtApp=app,
    #        mainfig=win_2,
    #        show_immediately=False)

    #win_3 = pg.GraphicsWindow(title="3")
    #Lineplot(DataModel_to_x_and_y_attr=[(statemon_synapse[41], ('t', 'Apre')), (statemon_synapse[49], ('t', 'Apost'))],
    #        title="Apost",
    #        xlabel='time',  # delta t
    #        ylabel='Apost',
    #        backend=visualization_backend,
    #        QtApp=app,
    #        mainfig=win_3,
    #        show_immediately=False)

    #win_4 = pg.GraphicsWindow(title="4")
    #Lineplot(DataModel_to_x_and_y_attr=[(statemon_synapse[25], ('t', 'Apre')), (statemon_synapse[25], ('t', 'Apost'))],
    #        title="Apost",
    #        xlabel='time',  # delta t
    #        ylabel='Apost',
    #        backend=visualization_backend,
    #        QtApp=app,
    #        mainfig=win_4,
    #        show_immediately=False)

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
