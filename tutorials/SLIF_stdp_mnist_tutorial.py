# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-01-16 17:57:35

"""
This file provides an example of how to use neuron and synapse models which are present
on neurmorphic chips in the context of synaptic plasticity based on precise timing of spikes.
We use a standard STDP protocal with a exponentioally decaying window.

"""
import pyqtgraph as pg
import numpy as np
import os
from struct import unpack

from brian2 import ms, Hz, ohm, mV, us, second, pA, prefs,\
    SpikeMonitor, StateMonitor, defaultclock,\
    implementation, check_units, ExplicitStateUpdater,\
    PoissonGroup

from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import StochasticLIF as neuron_model
from teili.models.synapse_models import StochasticSyn_decay_stoch_stdp as synapse_model
from teili.stimuli.testbench import STDP_Testbench
from teili.tools.add_run_reg import add_lfsr

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataModels import StateVariablesModel
from teili.tools.visualizer.DataControllers import Lineplot, Rasterplot

MNIST_data_path = '/home/pablo/git/stdp-mnist-brian2/data/'

def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    if bTrain:
        images = open(MNIST_data_path + 'train-images-idx3-ubyte','rb')
        labels = open(MNIST_data_path + 'train-labels-idx1-ubyte','rb')
    else:
        images = open(MNIST_data_path + 't10k-images-idx3-ubyte','rb')
        labels = open(MNIST_data_path + 't10k-labels-idx1-ubyte','rb')
    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = unpack('>I', images.read(4))[0]
    rows = unpack('>I', images.read(4))[0]
    cols = unpack('>I', images.read(4))[0]
    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = unpack('>I', labels.read(4))[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')
    # Get the data
    x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
    for i in range(10):#range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
        x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
        y[i] = unpack('>B', labels.read(1))[0]

    data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}

    return data

training = get_labeled_data(MNIST_data_path + 'training')
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
num_examples = 2
labels = [0] * num_examples
digits = [0] * num_examples
for i in range(num_examples):
    labels[i] = training['y'][i][0]
    digits[i] = training['x'][i]

n_input = 28 * 28
input_intensity = 2.
norm_factor = 8

input_groups = [PoissonGroup(n_input, 0*Hz) for _ in range(num_examples)]
for i, digit in enumerate(digits):
    # Scale rate as desired
    rate = digit.reshape(n_input) / norm_factor *  input_intensity
    input_groups[i].rates = rate * Hz

prefs.codegen.target = "numpy"
defaultclock.dt = 1 * ms
Net = TeiliNetwork()

stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')
post_neurons = Neurons(1,
                       equation_builder=neuron_model(num_inputs=1),
                       method=stochastic_decay,
                       name='post_neurons',
                       verbose=True)

stdp_synapse = Connections(input_groups[0], post_neurons,
                           method=stochastic_decay,
                           equation_builder=synapse_model(),
                           name='stdp_synapse')

stdp_synapse.connect(True)

# Set parameters:
seed = 12
add_lfsr(post_neurons, seed, defaultclock.dt)
post_neurons.Vm = 3*mV

stdp_synapse.tau_syn = 5*ms
add_lfsr(stdp_synapse, seed, defaultclock.dt)

# Setting up monitors
statemon_post_neurons = StateMonitor(post_neurons, variables=['Vm', 'Iin'],
        record=0, name='statemon_post_neurons')
spikemon_post_neurons = SpikeMonitor(post_neurons, name='spikemon_post_neurons')

statemon_post_synapse = StateMonitor(stdp_synapse, variables=['I_syn', 'w_plast'],
    record=True, name='statemon_post_synapse')

Net.add(post_neurons, statemon_post_neurons, input_groups[0], stdp_synapse,
        statemon_post_synapse, spikemon_post_neurons)

duration = 400
Net.run(duration * ms)

win_stdp = pg.GraphicsWindow(title="STDP Unit Test")
p1 = win_stdp.addPlot()
win_stdp.nextRow()
p2 = win_stdp.addPlot()
win_stdp.nextRow()
p3 = win_stdp.addPlot()
p2.setXLink(p1)
p3.setXLink(p2)

Lineplot(DataModel_to_x_and_y_attr=[(statemon_post_neurons, ('t', 'Vm'))],
            MyPlotSettings=PlotSettings(colors=['g']),
            x_range=(0, duration*1e-3),
            title="Membrane potential",
            xlabel="Time (ms)",
            ylabel="Vm (V)",
            backend='pyqtgraph',
            mainfig=win_stdp,
            subfig=p1)

Rasterplot(MyEventsModels=[spikemon_post_neurons],
           MyPlotSettings=PlotSettings(colors=['w']),
           time_range=(0, duration*1e-3),
           neuron_id_range=(-1, 1),
           xlabel="Time (ms)",
           ylabel="Neuron ID",
           backend='pyqtgraph',
           mainfig=win_stdp,
           subfig_rasterplot=p2)

Lineplot(DataModel_to_x_and_y_attr=[(statemon_post_neurons, ('t', 'Iin'))],
            MyPlotSettings=PlotSettings(colors=['g']),
            x_range=(0, duration*1e-3),
            title="Plastic synaptic weight",
            xlabel="Time (ms)",
            ylabel="Iin",
            backend='pyqtgraph',
            mainfig=win_stdp,
            subfig=p3,
            show_immediately=True)

# Visualize
img = np.array(input_groups[0].rates).reshape((28,28))
pg.image(img)

img = np.transpose(statemon_post_synapse.w_plast).reshape((duration, 28, 28))
pg.image(img)
