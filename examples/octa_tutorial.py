#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:45:02 2019

@author: Matteo
"""
import numpy as np
import matplotlib.pyplot as plt 

from brian2 import ms
from teili import TeiliNetwork
from teili.building_blocks.octa import Octa
from teili.models.parameters.octa_params import *
from teili.models.neuron_models import OCTA_Neuron as octa_neuron
from teili.stimuli.testbench import OCTA_Testbench
from teili.tools.sorting import SortMatrix

'''
Minimal working example for the OCTA network.

All documentation can be found on the teili website or in the octa.py docstrings.

The network parameters are found in teili.models.parameters.octa_params

'''

def plot_sorted_compression(OCTA):
    '''
    Plot the spiking activity of the compression layer sorted by similarity index.
    The out-of-the-box network the spiking activity should align with the input timing
    '''

    weights = OCTA.sub_blocks['compression'].groups['s_exc_exc'].w_plast
    s = SortMatrix(nrows=49, ncols=49, matrix=weights, axis=1)
    monitor = OCTA.sub_blocks['compression'].monitors['spikemon_exc']
    moni = np.asarray([np.where(np.asarray(s.permutation) == int(i))[0][0] for i in monitor.i])
    plt.figure(1)
    plt.plot(monitor.t, moni, '.r')
    plt.xlabel("Time")
    plt.ylabel("Sorted spikes")
    plt.title("Rasterplot compression block")


if __name__ == '__main__':
    Net = TeiliNetwork()

    OCTA_net = Octa(name='OCTA_net')

    testbench_stim = OCTA_Testbench()

    testbench_stim.rotating_bar(length=10, nrows=10,
                                direction='cw',
                                ts_offset=3, angle_step=10,
                                noise_probability=0.2,
                                repetitions=100,
                                debug=False)

    OCTA_net.groups['spike_gen'].set_spikes(indices=testbench_stim.indices,
                                            times=testbench_stim.times * ms)

    Net.add(OCTA_net, OCTA_net.sub_blocks['compression'],
            OCTA_net.sub_blocks['prediction'])

    Net.run(np.max(testbench_stim.times)* ms,
            report='text')

    plot_sorted_compression(OCTA=OCTA_net)


