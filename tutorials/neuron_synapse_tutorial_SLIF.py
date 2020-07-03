# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2017-25-08 13:43:10
"""
This is a tutorial to construct a simple network of neurons
using the teili framework.
The emphasise is on neuron groups and non-plastic synapses.
"""

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import numpy as np
import sys

from brian2 import ms, second, pA, nA, prefs,\
    SpikeMonitor, StateMonitor, \
    SpikeGeneratorGroup

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import StochasticLIF as neuron_model
from teili.models.synapse_models import StochasticSyn_decay as synapse_model
from teili.models.parameters.dpi_neuron_param import parameters as neuron_model_param

from teili.tools.visualizer.DataViewers import PlotSettings
from teili.tools.visualizer.DataControllers import Rasterplot, Lineplot

defaultclock.dt = 1*ms

@implementation('numpy', discard_units=True)
@check_units(decay_probability=1, num_neurons=1, lfsr_num_bits=1, result=1)
def lfsr(decay_probability, num_neurons, lfsr_num_bits):
    """
    Generate a pseudorandom number between 0 and 1 with a 20-bit Linear
    Feedback Shift Register (LFSR). This is equivalent to generating random
    numbers from an uniform distribution.

    This function receives a given number and performs num_neurons iterations
    of the LFSR. This is done to set the next input that will be used when a
    given neuron needs another random number. The LFSR does a circular shift
    (i.e. all the values are shifted left while the previous MSB becomes the
    new LSB) and ensures the variable is no bigger than 20 bits. After that,
    the 3rd bit is update with the result of a XOR between bits 3 and 0. Note
    that, for convenience, the input and outputs are normalized, i.e.
    value/2**20.

    Parameters
    ----------
    decay_probability : float
        Value between 0 and 1 that will be the input to the LFSR
    num_neurons : int
        Number of neurons in the group
    lfsr_num_bits : int
        Number of bits of the LFSR

    Returns
    -------
    float
        A random number between 0 and 1

    Examples
    --------
    >>> number = 2**19 + 2**2
    >>> bin(number)
    '0b10000000000000000100'
    >>> bin(int(lfsr(number/2**20, 1)*2**20))
    '0b1'
    """
    lfsr_num_bits = int(lfsr_num_bits[0])
    decay_probability *= 2**lfsr_num_bits
    #import pdb;pdb.set_trace()
    decay_probability = list(decay_probability)
    mask = 2**lfsr_num_bits - 1

    for i in range(num_neurons):
        decay_probability[i] = int(decay_probability[i]) << 1
        overflow = True if decay_probability[i] & (1 << lfsr_num_bits) else False
        # Re-introduces 1s beyond last position
        if overflow:
            decay_probability[i] |= 1
        # Ensures variable is lfsr_num_bits long
        decay_probability[i] = decay_probability[i] & mask
        # Get bits from positions 0 and 3
        fourth_tap = 1 if decay_probability[i] & (1 << 3) else 0
        first_tap = 1 if decay_probability[i] & (1 << 0) else 0
        # Update bit in position 3
        decay_probability[i] &=~ (1 << 3)
        if bool(fourth_tap^first_tap):
            decay_probability[i] |= (1 << 3)

    return np.array(decay_probability)/2**lfsr_num_bits

def init_lfsr(lfsr_seed, num_neurons, num_bits):
    """
    Initializes numbers that will be used for each neuron on the LFSR
    function by iterating on the LFSR num_neurons times.

    Parameters
    ----------
    lfsr_seed : int
        The seed of the LFSR
    num_neurons : int
        Number of neurons in the group
    num_bits : int
        Number of bits of the LFSR

    Returns
    -------
    lfsr_out : numpy.array of float
        The initial values of each neuron

    """
    lfsr_out = [0 for _ in range(num_neurons)]
    mask = 2**num_bits - 1

    for i in range(num_neurons):
        lfsr_seed = lfsr_seed << 1
        overflow = True if lfsr_seed & (1 << num_bits) else False

        # Re-introduces 1s beyond last position
        if overflow:
            lfsr_seed |= 1

        # Ensures variable is num_bits long
        lfsr_seed = lfsr_seed & mask

        # Get bits from positions 0 and 3
        fourth_tap = 1 if lfsr_seed & (1 << 3) else 0
        first_tap = 1 if lfsr_seed & (1 << 0) else 0
        # Update bit in position 3
        lfsr_seed &=~ (1 << 3)
        if bool(fourth_tap^first_tap):
            lfsr_seed |= (1 << 3)
        lfsr_out[i] = lfsr_seed

    return np.asarray(lfsr_out)/2**num_bits

prefs.codegen.target = "numpy"

input_timestamps = np.asarray([1, 3, 4, 5, 6, 7, 8, 9]) * ms
input_indices = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])
input_spikegenerator = SpikeGeneratorGroup(1, indices=input_indices,
                                           times=input_timestamps, 
                                           name='input_spikegenerator')


Net = TeiliNetwork()

test_neurons1 = Neurons(N=2, 
                        equation_builder=neuron_model(num_inputs=2), 
                        name="test_neurons1",
                        verbose=True)

test_neurons2 = Neurons(N=2, 
                        equation_builder=neuron_model(num_inputs=2), 
                        name="test_neurons2",
                        verbose=True)

input_synapse = Connections(input_spikegenerator, test_neurons1,
                            equation_builder=synapse_model(), 
                            name="input_synapse", verbose=True)
input_synapse.connect(True)

test_synapse = Connections(test_neurons1, test_neurons2,
                           equation_builder=synapse_model(), name="test_synapse")
test_synapse.connect(True)

'''
You can change all the parameters like this after creation
of the neurongroup or synapsegroup.
Note that the if condition is inly there for
convinience to switch between voltage- or current-based models.
Normally, you have one or the other in yur simulation, thus
you will not need the if condition.
'''
lfsr_seed = 12345
num_bits = int(test_neurons1.lfsr_num_bits[0])
num_elements = len(test_neurons1.lfsr_num_bits)
test_neurons1.decay_probability = init_lfsr(lfsr_seed, num_elements, num_bits)
test_neurons1.namespace.update({'lfsr': lfsr})
test_neurons1.Vm = 3*mV
test_neurons1.run_regularly('''decay_probability = lfsr(decay_probability,\
                                                 N,\
                                                 lfsr_num_bits)
                     ''',
                     dt=defaultclock.dt)

num_bits = int(input_synapse.lfsr_num_bits_syn[0])
num_elements = len(input_synapse.lfsr_num_bits_syn)
input_synapse.psc_decay_probability = init_lfsr(lfsr_seed, test_neurons1.N, num_bits)
input_synapse.namespace.update({'lfsr': lfsr})
input_synapse.run_regularly('''psc_decay_probability = lfsr(psc_decay_probability,\
                                                      N,\
                                                      lfsr_num_bits_syn)
                      ''',
                      dt=defaultclock.dt)

num_bits = int(test_synapse.lfsr_num_bits_syn[0])
num_elements = len(test_synapse.lfsr_num_bits_syn)
test_synapse.psc_decay_probability = init_lfsr(lfsr_seed, num_elements, num_bits)
test_synapse.namespace.update({'lfsr': lfsr})
test_synapse.run_regularly('''psc_decay_probability = lfsr(psc_decay_probability,\
                                                      N,\
                                                      lfsr_num_bits_syn)
                      ''',
                      dt=defaultclock.dt)

# Example of how to set parameters, saved as a dictionary
test_neurons1.set_params(neuron_model_param)
# Example of how to set a single parameter
test_neurons1.refP = 1 * ms
test_neurons2.set_params(neuron_model_param)
test_neurons2.refP = 1 * ms
if 'Imem' in neuron_model().keywords['model']:
    input_synapse.weight = 5000
    test_synapse.weight = 800
    test_neurons1.Iconst = 10 * nA
elif 'Vm' in neuron_model().keywords['model']:
    input_synapse.weight = 1.5
    test_synapse.weight = 8.0
    test_neurons1.Iconst = 10000000 * nA

spikemon_input = SpikeMonitor(input_spikegenerator, name='spikemon_input')
spikemon_test_neurons1 = SpikeMonitor(
    test_neurons1, name='spikemon_test_neurons1')
spikemon_test_neurons2 = SpikeMonitor(
    test_neurons2, name='spikemon_test_neurons2')

statemon_input_synapse = StateMonitor(
    input_synapse, variables='I_syn', record=True, name='statemon_input_synapse')

statemon_test_synapse = StateMonitor(
    test_synapse, variables='I_syn', record=True, name='statemon_test_synapse')

if 'Imem' in neuron_model().keywords['model']:
    statemon_test_neurons2 = StateMonitor(test_neurons2,
                                          variables=['Imem'],
                                          record=0, name='statemon_test_neurons2')
    statemon_test_neurons1 = StateMonitor(test_neurons1, variables=[
        "Iin", "Imem", "Iahp"], record=[0, 1], name='statemon_test_neurons1')
elif 'Vm' in neuron_model().keywords['model']:
    statemon_test_neurons2 = StateMonitor(test_neurons2,
                                          variables=['Vm'],
                                          record=0, name='statemon_test_neurons2')
    statemon_test_neurons1 = StateMonitor(test_neurons1, variables=[
        "Iin", "Vm", "Iadapt"], record=[0, 1], name='statemon_test_neurons1')


Net.add(input_spikegenerator, test_neurons1, test_neurons2,
        input_synapse, test_synapse,
        spikemon_input, spikemon_test_neurons1, spikemon_test_neurons2,
        statemon_test_neurons1, statemon_test_neurons2,
        statemon_test_synapse, statemon_input_synapse)

duration = 0.5
Net.run(duration * second)

# Visualize simulation results
app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))

pg.setConfigOptions(antialias=True)
labelStyle = {'color': '#FFF', 'font-size': 12}
MyPlotSettings = PlotSettings(fontsize_title=labelStyle['font-size'],
                              fontsize_legend=labelStyle['font-size'],
                              fontsize_axis_labels=10,
                              marker_size=7)

win = pg.GraphicsWindow()
win.resize(2100, 1200)
win.setWindowTitle('Simple Spiking Neural Network')

p1 = win.addPlot(title="Input spike generator")
p2 = win.addPlot(title="Input synapses")
win.nextRow()
p3 = win.addPlot(title='Intermediate test neurons 1')
p4 = win.addPlot(title="Test synapses")
win.nextRow()
p5 = win.addPlot(title="Rasterplot of output test neurons 2")
p6 = win.addPlot(title="Output test neurons 2")


# Spike generator
Rasterplot(MyEventsModels=[spikemon_input],
           MyPlotSettings=MyPlotSettings,
           time_range=[0, duration],
           neuron_id_range=None,
           title="Input spike generator",
           xlabel='Time (ms)',
           ylabel="Neuron ID",
           backend='pyqtgraph',
           mainfig=win,
           subfig_rasterplot=p1,
           QtApp=app,
           show_immediately=False)

# Input synapses
Lineplot(DataModel_to_x_and_y_attr=[(statemon_input_synapse, ('t', 'I_syn'))],
         MyPlotSettings=MyPlotSettings,
         x_range=[0, duration],
         title="Input synapses",
         xlabel="Time (ms)",
         ylabel="EPSC (A)",
         backend='pyqtgraph',
         mainfig=win,
         subfig=p2,
         QtApp=app,
         show_immediately=False)

# Intermediate neurons
if hasattr(statemon_test_neurons1, 'Imem'):
    MyData_intermed_neurons = [(statemon_test_neurons1, ('t', 'Imem'))]
if hasattr(statemon_test_neurons1, 'Vm'):
    MyData_intermed_neurons = [(statemon_test_neurons1, ('t', 'Vm'))]

i_current_name = 'Imem' if 'Imem' in neuron_model().keywords['model'] else 'Vm'
Lineplot(DataModel_to_x_and_y_attr=MyData_intermed_neurons,
         MyPlotSettings=MyPlotSettings,
         x_range=[0, duration],
         title='Intermediate test neurons 1',
         xlabel="Time (ms)",
         ylabel=i_current_name,
         backend='pyqtgraph',
         mainfig=win,
         subfig=p3,
         QtApp=app,
         show_immediately=False)

# Output synapses
Lineplot(DataModel_to_x_and_y_attr=[(statemon_test_synapse, ('t', 'I_syn'))],
         MyPlotSettings=MyPlotSettings,
         x_range=[0, duration],
         title="Test synapses",
         xlabel="Time (ms)",
         ylabel="EPSC (A)",
         backend='pyqtgraph',
         mainfig=win,
         subfig=p4,
         QtApp=app,
         show_immediately=False)


Rasterplot(MyEventsModels=[spikemon_test_neurons2],
           MyPlotSettings=MyPlotSettings,
           time_range=[0, duration],
           neuron_id_range=None,
           title="Rasterplot of output test neurons 2",
           xlabel='Time (ms)',
           ylabel="Neuron ID",
           backend='pyqtgraph',
           mainfig=win,
           subfig_rasterplot=p5,
           QtApp=app,
           show_immediately=False)

if hasattr(statemon_test_neurons2, 'Imem'):
    MyData_output = [(statemon_test_neurons2, ('t', 'Imem'))]
if hasattr(statemon_test_neurons2, 'Vm'):
    MyData_output = [(statemon_test_neurons2, ('t', 'Vm'))]

Lineplot(DataModel_to_x_and_y_attr=MyData_output,
         MyPlotSettings=MyPlotSettings,
         x_range=[0, duration],
         title="Output test neurons 2",
         xlabel="Time (ms)",
         ylabel="%s" % i_current_name,
         backend='pyqtgraph',
         mainfig=win,
         subfig=p6,
         QtApp=app,
         show_immediately=False)

app.exec()
