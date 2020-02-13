#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Thu Jun 27 14:45:02 2019

# @author: Matteo
""" Minimal working example for an OCTA (Online Clustering of Temporal Activity) network.
Detailed documentation can be found on our documentation or directly
in the docstrings of the `building_block` located in `teili/building_blocks/octa.py`.

The network's parameters can be found in `teili/models/parameters/octa_params.py`.
"""
import numpy as np
import matplotlib.pyplot as plt 
from brian2 import us, ms, prefs, defaultclock, core, float64
import copy as cp

from teili import TeiliNetwork
from teili.building_blocks.octa import Octa
from teili.models.parameters.octa_params import wta_params, octa_params,\
    mismatch_neuron_param, mismatch_synap_param
from teili.models.neuron_models import OCTA_Neuron as octa_neuron
from teili.stimuli.testbench import OCTA_Testbench
from teili.tools.sorting import SortMatrix

def plot_sorted_compression(OCTA_net):
    """ Plot the spiking activity, i.e. spike rasterplot of the compression
    layer, sorted by similarity. Similarity is calculated based on euclidean
    distance. The sorting yields permuted indices which are used to re-order the
    neurons.
    The out-of-the-box network with the default testbench stimulus will show an
    alignment with time, such that the sorted spike rasterplot can be fitted by
    a line with increasing or decreasing neuron index. As the starting neuron is
    picked at random the alignment with time can vary between runs.

    Arguments:
        OCTA (TeiliNetwork): The `TeiliNetwork` which contains the OCTA `BuildingBlock`. 
    """
    
    weights = cp.deepcopy(np.asarray(OCTA_net.sub_blocks['compression'].groups['s_exc_exc'].w_plast))
    indices = cp.deepcopy(np.asarray(OCTA_net.sub_blocks['compression'].monitors['spikemon_exc'].i))
    time = cp.deepcopy(np.asarray(OCTA_net.sub_blocks['compression'].monitors['spikemon_exc'].t))

    s = SortMatrix(nrows=49, ncols=49, matrix=weights, axis=1)
    # We use the permuted indices to sort the neuron ids
    sorted_ind = np.asarray([np.where(np.asarray(s.permutation) == int(i))[0][0] for i in indices])

    plt.figure(1)
    plt.plot(time, sorted_ind, '.r')
    plt.xlabel('Time')
    plt.ylabel('Sorted spikes')
    plt.title('Rasterplot compression block')
    plt.show()


if __name__ == '__main__':
    prefs.codegen.target = "numpy"
    defaultclock.dt = 500 * us
    core.default_float_dtype = float64

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

    Net.add(OCTA_net,
            OCTA_net.sub_blocks['compression'],
            OCTA_net.sub_blocks['prediction'])
    Net.run(np.max(testbench_stim.times) * ms,
            report='text')

    plot_sorted_compression(OCTA_net)

