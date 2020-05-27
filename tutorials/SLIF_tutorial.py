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

# TODO modular functions, n_bits as parameter
@implementation('numpy', discard_units=True)
@check_units(lfsr_in=1, num_neurons=1, result=1)
def LFSR_20bit(lfsr_in, num_neurons):
    """
    Generate pseudorandom number with a 20-bit Linear Feedback Shift Register
    (LFSR) for num_neurons neurons.

    This function receives a given number and performs num_neurons iterations
    of the LFSR. The LFSR does a circular shift (i.e. all the values are
    shifted left while the previous MSB becomes the new LSB) and ensures the
    variable is no bigger than 20 bits. After that, the 3rd bit is update with
    the result of a XOR between bits 3 and 0. Note that, for convenience, the
    input and outputs are normalized, i.e. value/2**20.

    Parameters
    ----------
    lfsr_in : float
        Value between 0 and 1 that will be the input to the LFSR
    num_neurons : int
        Number of neurons in the group

    Returns
    -------
    lfsr_out : float
        Output of the LFSR

    Examples
    --------
    >>> number = 2**19 + 2**2
    >>> bin(number)
    '0b10000000000000000100'
    >>> bin(int(LFSR_20bit(number/2**20, 1)2**20))
    '0b1'
    """
    num_bits = 20
    lfsr_in *= 2**20
    for _ in range(num_neurons):
        lfsr_in = int(lfsr_in) << 1
        overflow = True if lfsr_in & (1 << num_bits) else False
        # Re-introduces 1s beyond last position
        if overflow:
            lfsr_in |= 1
        # Ensures variable is num_bits long
        mask = 2**num_bits - 1
        lfsr_in = lfsr_in & mask
        # Get bits from positions 0 and 3
        fourth_tap = 1 if lfsr_in & (1 << 3) else 0
        first_tap = 1 if lfsr_in & (1 << 0) else 0
        # Update bit in position 3
        lfsr_in &=~ (1 << 3)
        if bool(fourth_tap^first_tap):
            lfsr_in |= (1 << 3)

    lfsr_out = lfsr_in/2**20

    return lfsr_out

def init_lfsr(lfsr_seed, num_neurons):
    """
    Initializes numbers that will be used for each neuron on the LFSR
    function by iterating on the LFSR num_neurons times.

    Parameters
    ----------
    lfsr_seed : float
        Value between 0 and 1 that will be the input to the LFSR
    num_neurons : int
        Number of neurons in the group

    Returns
    -------
    lfsr_out : float
        Output of the LFSR

    """
    num_bits = 20
    lfsr_out = [0 for _ in range(num_neurons)]
    # Set initial value from provided seed, as in barebone python code
    lfsr_tmp = int(lfsr_seed)

    for i in range(num_neurons):
        lfsr_tmp = lfsr_tmp << 1
        overflow = True if lfsr_tmp & (1 << num_bits) else False

        # Re-introduces 1s beyond last position
        if overflow:
            lfsr_tmp |= 1
        
        # Ensures variable is num_bits long
        mask = 2**num_bits - 1
        lfsr_tmp = lfsr_tmp & mask

        # Get bits from positions 0 and 3
        fourth_tap = 1 if lfsr_tmp & (1 << 3) else 0
        first_tap = 1 if lfsr_tmp & (1 << 0) else 0
        # Update bit in position 3
        lfsr_tmp &=~ (1 << 3)
        if bool(fourth_tap^first_tap):
            lfsr_tmp |= (1 << 3)
        lfsr_out[i] = lfsr_tmp

    return np.asarray(lfsr_out)/2**20

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

neuron.Vmem = 3
lfsr_seed = 12345
neuron.lfsr_input = init_lfsr(lfsr_seed, neuron.N)
neuron.namespace.update({'LFSR_20bit': LFSR_20bit})
neuron.run_regularly('''lfsr_input = LFSR_20bit(lfsr_input, N)
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

#import pdb; pdb.set_trace()
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
