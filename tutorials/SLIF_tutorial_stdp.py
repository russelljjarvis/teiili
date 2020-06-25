# -*- coding: utf-8 -*-
# @Author: pabloabur
# @Date:   2020-20-06 18:05:05
"""
This is a simple tutorial to simulate 4 stochastic leaky integrate and fire
neurons charging and firing action potentials with STDP. Inputs provided are
digits from MNIST dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack

from brian2 import ExplicitStateUpdater
from brian2 import StateMonitor, SpikeMonitor, ms, mV, Hz, defaultclock,\
        implementation, check_units, SpikeGeneratorGroup, PoissonGroup

from teili import TeiliNetwork
from teili.core.groups import Neurons, Connections
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder

defaultclock.dt = 1*ms

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
    lfsr_num_bits = int(lfsr_num_bits)
    decay_probability *= 2**lfsr_num_bits
    mask = 2**lfsr_num_bits - 1

    for _ in range(num_neurons):
        decay_probability = int(decay_probability) << 1
        overflow = True if decay_probability & (1 << lfsr_num_bits) else False
        # Re-introduces 1s beyond last position
        if overflow:
            decay_probability |= 1
        # Ensures variable is lfsr_num_bits long
        decay_probability = decay_probability & mask
        # Get bits from positions 0 and 3
        fourth_tap = 1 if decay_probability & (1 << 3) else 0
        first_tap = 1 if decay_probability & (1 << 0) else 0
        # Update bit in position 3
        decay_probability &=~ (1 << 3)
        if bool(fourth_tap^first_tap):
            decay_probability |= (1 << 3)

    return decay_probability/2**lfsr_num_bits

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

path = os.path.expanduser("~")
model_path = os.path.join(path, "teiliApps", "equations", "")

neuron_model = NeuronEquationBuilder.import_eq(
    filename=model_path + 'StochasticLIF.py', num_inputs=2)
synapse_model = SynapseEquationBuilder.import_eq(
    filename=model_path + 'StochasticLIFSyn.py')

# Define inputs
training = get_labeled_data(MNIST_data_path + 'training')
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
num_examples = 2
labels = [0] * num_examples
digits = [0] * num_examples
for i in range(num_examples):
    labels[i] = training['y'][i][0]
    digits[i] = training['x'][i]
#    print(f'Sample {i}: {labels[i]}')
#    plt.figure()
#    plt.imshow(digits[i], cmap='gray')
#plt.show()

n_e = 400
n_input = 28 * 28
input_intensity = 2.

input_groups = [PoissonGroup(n_input, 0*Hz) for _ in range(num_examples)]
for i, digit in enumerate(digits):
    # Scale rate as desired
    rate = digit.reshape(n_input) / 8. *  input_intensity
    input_groups[i].rates = rate * Hz

duration = 20*ms
input_monitor = SpikeMonitor(input_groups[0])
Net = TeiliNetwork()
Net.add(input_monitor, input_groups[0])
Net.run(duration)
input_timestamps = input_monitor.t
input_indices = input_monitor.i
#input_timestamps = np.array(range(1, 400, 100))*ms
#input_indices = np.zeros(len(input_timestamps))
input_spike_generator = SpikeGeneratorGroup(n_input, indices=input_indices,
                                            times=input_timestamps)

num_neurons = 4
# With this state updater, the abstract state update code generated by Brian2
# will be the same as the equation provided
stochastic_decay = ExplicitStateUpdater('''x_new = dt*f(x,t)''')
neuron = Neurons(num_neurons, equation_builder=neuron_model,
                 method=stochastic_decay, name="test_neurons", verbose=True)

synapse = Connections(input_spike_generator, neuron, method=stochastic_decay,
                      equation_builder=synapse_model, verbose=True)
synapse.connect(True)

lfsr_seed = 12345
lfsr_num_bits = 20
lfsr_seed = 12345
lfsr_num_bits = 20
neuron.add_state_variable('lfsr_num_bits', shared=True, constant=True)
neuron.lfsr_num_bits = lfsr_num_bits
neuron.decay_probability = init_lfsr(lfsr_seed, neuron.N, lfsr_num_bits)
neuron.namespace.update({'lfsr': lfsr})
neuron.Vm = 3*mV
neuron.run_regularly('''decay_probability = lfsr(decay_probability,\
                                                 N,\
                                                 lfsr_num_bits)
                     ''',
                     dt=1*ms)

synapse.weight = 0.1
synapse.add_state_variable('lfsr_num_bits_syn', shared=True, constant=True)
lfsr_num_bits_syn = 20
synapse.lfsr_num_bits_syn = lfsr_num_bits_syn
#import pdb; pdb.set_trace()
# TODO sending synapse.N results in an error with VariableView not being integer
synapse.psc_decay_probability = init_lfsr(lfsr_seed, len(synapse.N_incoming), lfsr_num_bits_syn)
synapse.namespace.update({'lfsr': lfsr})
synapse.run_regularly('''psc_decay_probability = lfsr(psc_decay_probability,\
                                                      N,\
                                                      lfsr_num_bits_syn)
                      ''',
                      dt=1*ms)

spikemon = SpikeMonitor(neuron, name='spike_monitor')
neuron_monitor = StateMonitor(neuron, variables=['Vm', 'Iin', 'decay_probability'], record=True, name='state_monitor_neu')
synapse_monitor = StateMonitor(synapse, variables=['I_syn'], record=True, name='state_monitor_syn')

Net = TeiliNetwork()
Net.add(input_spike_generator, neuron, synapse, spikemon, neuron_monitor, synapse_monitor)
Net.run(duration, report='text')

# Visualize compare random numbers generated with rand, for 250s
#plt.figure()
#Z = np.random.rand(250000)   # Test data
#Z = np.reshape(Z, (500,500))
#plt.imshow(Z, cmap='gray', interpolation='nearest')
#plt.show()
#for i in range(num_neurons):
#    plt.figure()
#    Z = neuron_monitor.decay_probability[i,:]
#    Z = np.reshape(Z, (500,500))
#    plt.imshow(Z, cmap='gray', interpolation='nearest')
#    plt.show()

#plt.figure()
#spike_times = spikemon[0].t/ms
#ISI = spike_times[1:-1] - spike_times[0:-2]
#_ = plt.hist(ISI, bins=20, range=(0, 100))

plt.figure()
plt.plot(spikemon.t/ms, spikemon.i, 'ko')
plt.title('Neurons raster')
plt.ylabel('Neuron index')
plt.xlabel('Time (ms)')

plt.figure()
plt.plot(input_timestamps/ms, input_indices, 'ko')
plt.title('Input poissonian spikes')
plt.ylabel('Input index')
plt.xlabel('Time (ms)')

plt.figure()
colors = ['k', 'r', 'g', 'b']
for i in range(num_neurons):
    plt.plot(neuron_monitor.t/ms, neuron_monitor.Vm[i], colors[i], label=str(i))
plt.title('Membrane potential of the 4 neurons')
plt.xlabel('Time (ms)')
plt.ylabel('membrane potential (V)')
plt.legend()

plt.figure()
for i in range(num_neurons):
    plt.plot(neuron_monitor.t/ms, neuron_monitor.Iin[i], 'k')
plt.title('Input current')
plt.xlabel('Time (ms)')
plt.ylabel('Iin (A)');

plt.figure()
for i in range(num_neurons):
    plt.plot(synapse_monitor.t/ms, synapse_monitor.I_syn[i], 'k')
plt.title('Input current')
plt.xlabel('Time (ms)')
plt.ylabel('I_syn (A)');
plt.show()

#with open('teili.npy', 'wb') as f:
#    np.save(f, neuron_monitor.Vm)
