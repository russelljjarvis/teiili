# -*- coding: utf-8 -*-
# @Author: pabloabur
# @Date:   2020-27-05 18:05:05
"""
This is a simple tutorial to simulate 4 stochastic leaky integrate and fire
neurons charging and firing action potentials.
"""

import os
import numpy as np

from brian2 import ExplicitStateUpdater
import matplotlib.pyplot as plt

from brian2 import StateMonitor, SpikeMonitor, ms, defaultclock,\
        implementation, check_units
from teili import TeiliNetwork
from teili.core.groups import Neurons
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

defaultclock.dt = 1*ms

@implementation('numpy', discard_units=True)
@check_units(lfsr_in=1, num_neurons=1, num_bits=1, result=1)
def LFSR_gen(lfsr_in, num_neurons, num_bits):
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
    lfsr_in : float
        Value between 0 and 1 that will be the input to the LFSR
    num_neurons : int
        Number of neurons in the group
    num_bits : int
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
    >>> bin(int(LFSR_gen(number/2**20, 1)2**20))
    '0b1'
    """
    num_bits = int(num_bits)
    lfsr_in *= 2**num_bits
    mask = 2**num_bits - 1

    for _ in range(num_neurons):
        lfsr_in = int(lfsr_in) << 1
        overflow = True if lfsr_in & (1 << num_bits) else False
        # Re-introduces 1s beyond last position
        if overflow:
            lfsr_in |= 1
        # Ensures variable is num_bits long
        lfsr_in = lfsr_in & mask
        # Get bits from positions 0 and 3
        fourth_tap = 1 if lfsr_in & (1 << 3) else 0
        first_tap = 1 if lfsr_in & (1 << 0) else 0
        # Update bit in position 3
        lfsr_in &=~ (1 << 3)
        if bool(fourth_tap^first_tap):
            lfsr_in |= (1 << 3)

    return lfsr_in/2**num_bits

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

N = 4
duration = 400*ms

path = os.path.expanduser("~")
model_path = os.path.join(path, "teiliApps", "equations", "")

builder_object = NeuronEquationBuilder.import_eq(
    filename=model_path + 'StochasticLIF.py', num_inputs=2)

Net = TeiliNetwork()
# With this state updater, the equation used is the one provided
stochastic_decay = ExplicitStateUpdater('''x_new = dt*f(x,t)''')
neuron = Neurons(N, equation_builder=builder_object, method=stochastic_decay,
                 name="test_neurons", verbose=True)

lfsr_seed = 12345
lfsr_len = 20
neuron.add_state_variable('lfsr_len', shared=True, constant=True)
neuron.lfsr_len = lfsr_len
neuron.rand_input = init_lfsr(lfsr_seed, neuron.N, lfsr_len)
neuron.namespace.update({'LFSR_gen': LFSR_gen})

neuron.Vmem = 3
neuron.run_regularly('''rand_input = LFSR_gen(rand_input, N, lfsr_len)
                        ''',
                     dt=1*ms)

spikemon = SpikeMonitor(neuron, name='spike_monitor')
M = StateMonitor(neuron, variables=['Vmem'], record=True, name='state_monitor')

Net.add(neuron, spikemon, M)
Net.run(duration)

#plt.figure()
#spike_times = spikemon[0].t/ms
#ISI = spike_times[1:-1] - spike_times[0:-2]
#_ = plt.hist(ISI, bins=20, range=(0, 100))

plt.figure()
plt.plot(spikemon.t/ms, spikemon.i, 'ro')
plt.ylabel('Neuron index')
plt.xlabel('Time (samples)')

plt.figure()
for i in range(N):
    plt.plot(M.t/ms, M.Vmem[i], 'k')
plt.xlabel('Time (samples)')
plt.ylabel('membrane potential (a.u.)');
plt.show()

#with open('teili.npy', 'wb') as f:
#    np.save(f, M.Vmem)
