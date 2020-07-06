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

from brian2 import ms, mV, us, second, pA, prefs,\
    SpikeMonitor, StateMonitor, defaultclock,\
    implementation, check_units, ExplicitStateUpdater

from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn, DPIstdp
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
    for i in range(N):
        if i % 1000 == 0:
            print("i: %i" % i)
        x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
        y[i] = unpack('>B', labels.read(1))[0]

    data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}

    return data

path = os.path.expanduser("~")
model_path = os.path.join(path, "teiliApps", "equations", "")

neuron_model = NeuronEquationBuilder.import_eq(
    filename=model_path + 'StochasticLIF.py', num_inputs=2)
synapse_model = SynapseEquationBuilder.import_eq(
    filename=model_path + 'StochasticLIFSyn.py')

################
# Define MNIST inputs
################
#training = get_labeled_data(MNIST_data_path + 'training')
#testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
#num_examples = 2
#labels = [0] * num_examples
#digits = [0] * num_examples
#for i in range(num_examples):
#    labels[i] = training['y'][i][0]
#    digits[i] = training['x'][i]
##    print(f'Sample {i}: {labels[i]}')
##    plt.figure()
##    plt.imshow(digits[i], cmap='gray')
##plt.show()
#
#n_e = 400
#n_input = 28 * 28
#input_intensity = 2.
#
#input_groups = [PoissonGroup(n_input, 0*Hz) for _ in range(num_examples)]
#for i, digit in enumerate(digits):
#    # Scale rate as desired
#    rate = digit.reshape(n_input) / 8. *  input_intensity
#    input_groups[i].rates = rate * Hz
#
#duration = 20*ms
#input_monitor = SpikeMonitor(input_groups[0])
#Net2 = TeiliNetwork()
#Net2.add(input_monitor, input_groups[0])
#Net2.run(duration)
#input_timestamps = input_monitor.t
#input_indices = input_monitor.i
##input_timestamps = np.array(range(1, 400, 100))*ms
##input_indices = np.zeros(len(input_timestamps))
#input_spike_generator = SpikeGeneratorGroup(n_input, indices=input_indices,
#                                            times=input_timestamps)

prefs.codegen.target = "numpy"
defaultclock.dt = 50 * us
Net = TeiliNetwork()

stochastic_decay = ExplicitStateUpdater('''x_new = dt*f(x,t)''')
stdp = STDP_Testbench()
pre_spikegenerator, post_spikegenerator = stdp.stimuli(isi=30)

pre_neurons = Neurons(2, equation_builder=neuron_model, method=stochastic_decay,
                      name='pre_neurons', verbose=True)

post_neurons = Neurons(2, equation_builder=neuron_model, method=stochastic_decay,
                       name='post_neurons', verbose=True)


#synapse = Connections(input_spike_generator, neuron, method=stochastic_decay,
#                      equation_builder=synapse_model, verbose=True)
#synapse.connect(True)
pre_synapse = Connections(pre_spikegenerator, pre_neurons, method=stochastic_decay,
                          equation_builder=synapse_model, name='pre_synapse')

post_synapse = Connections(post_spikegenerator, post_neurons, method=stochastic_decay,
                           equation_builder=synapse_model, name='post_synapse')

stdp_synapse = Connections(pre_neurons, post_neurons,
                           equation_builder=DPIstdp(), name='stdp_synapse')

pre_synapse.connect(True)
post_synapse.connect(True)
# Set parameters:
pre_neurons.refP = 3 * ms
# TODO
#pre_neurons.Itau = 6 * pA
#post_neurons.Itau = 6 * pA

pre_synapse.weight = 4000.

post_synapse.weight = 4000.

stdp_synapse.connect("i==j")
stdp_synapse.weight = 300.
stdp_synapse.I_tau = 10 * pA
stdp_synapse.dApre = 0.01
stdp_synapse.taupre = 3 * ms
stdp_synapse.taupost = 3 * ms

add_lfsr(pre_neurons, 12345, defaultclock.dt)
pre_neurons.Vm = 3*mV
add_lfsr(post_neurons, 12345, defaultclock.dt)
post_neurons.Vm = 3*mV

pre_synapse.weight = 3
add_lfsr(pre_synapse, 12345, defaultclock.dt)
post_synapse.weight = 3
add_lfsr(post_synapse, 12345, defaultclock.dt)

# Setting up monitors
spikemon_pre_neurons = SpikeMonitor(pre_neurons, name='spikemon_pre_neurons')
statemon_pre_neurons = StateMonitor(pre_neurons, variables='Vm',# TODO variables='Imem', 
                                    record=0, name='statemon_pre_neurons')

spikemon_post_neurons = SpikeMonitor(
    post_neurons, name='spikemon_post_neurons')
statemon_post_neurons = StateMonitor(# TODO variables='Imem', 
    post_neurons, variables='Vm', record=0, name='statemon_post_neurons')


statemon_pre_synapse = StateMonitor(
    pre_synapse, variables=['I_syn'], record=0, name='statemon_pre_synapse')

statemon_post_synapse = StateMonitor(stdp_synapse, variables=[
    'I_syn', 'w_plast', 'weight'],
    record=True, name='statemon_post_synapse')

Net.add(pre_spikegenerator, post_spikegenerator,
        pre_neurons, post_neurons,
        pre_synapse, post_synapse, stdp_synapse,
        spikemon_pre_neurons, spikemon_post_neurons,
        statemon_pre_neurons, statemon_post_neurons,
        statemon_pre_synapse, statemon_post_synapse)

duration = 2.
Net.run(duration * second)


# Visualize
win_stdp = pg.GraphicsWindow(title="STDP Unit Test")
win_stdp.resize(2500, 1500)
win_stdp.setWindowTitle("Spike Time Dependent Plasticity")

p1 = win_stdp.addPlot()
win_stdp.nextRow()
p2 = win_stdp.addPlot()
win_stdp.nextRow()
p3 = win_stdp.addPlot()

text1 = pg.TextItem(text='Homoeostasis', anchor=(-0.3, 0.5))
text2 = pg.TextItem(text='Weak Pot.', anchor=(-0.3, 0.5))
text3 = pg.TextItem(text='Weak Dep.', anchor=(-0.3, 0.5))
text4 = pg.TextItem(text='Strong Pot.', anchor=(-0.3, 0.5))
text5 = pg.TextItem(text='Strong Dep.', anchor=(-0.3, 0.5))
text6 = pg.TextItem(text='Homoeostasis', anchor=(-0.3, 0.5))
p1.addItem(text1)
p1.addItem(text2)
p1.addItem(text3)
p1.addItem(text4)
p1.addItem(text5)
p1.addItem(text6)
text1.setPos(0, 0.5)
text2.setPos(0.300, 0.5)
text3.setPos(0.600, 0.5)
text4.setPos(0.900, 0.5)
text5.setPos(1.200, 0.5)
text6.setPos(1.500, 0.5)

Rasterplot(MyEventsModels=[spikemon_pre_neurons, spikemon_post_neurons],
            MyPlotSettings=PlotSettings(colors=['w', 'r']),
            time_range=(0, duration),
            neuron_id_range=(-1, 2),
            title="STDP protocol",
            xlabel="Time (s)",
            ylabel="Neuron ID",
            backend='pyqtgraph',
            mainfig=win_stdp,
            subfig_rasterplot=p1)

Lineplot(DataModel_to_x_and_y_attr=[(statemon_post_synapse, ('t', 'w_plast'))],
            MyPlotSettings=PlotSettings(colors=['g']),
            x_range=(0, duration),
            title="Plastic synaptic weight",
            xlabel="Time (s)",
            ylabel="Synpatic weight w_plast",
            backend='pyqtgraph',
            mainfig=win_stdp,
            subfig=p2)

datamodel = StateVariablesModel(state_variable_names=['I_syn'],
                                state_variables=[np.asarray(statemon_post_synapse.I_syn[1])],
                                state_variables_times=[np.asarray(statemon_post_synapse.t)])
Lineplot(DataModel_to_x_and_y_attr=[(datamodel, ('t_I_syn', 'I_syn'))],
            MyPlotSettings=PlotSettings(colors=['m']),
            x_range=(0, duration),
            title="Post synaptic current",
            xlabel="Time (s)",
            ylabel="Synapic current I (pA)",
            backend='pyqtgraph',
            mainfig=win_stdp,
            subfig=p3,
            show_immediately=True)
