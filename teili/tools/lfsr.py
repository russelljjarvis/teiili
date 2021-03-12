from brian2 import ms, TimedArray

from teili.core.groups import Neurons, Connections

import sys

import numpy as np


def get_parameters(n_elements, lfsr, prev_index):
    max_value = lfsr['max_value'][prev_index:prev_index+n_elements]*ms
    init = lfsr['init'][prev_index:prev_index+n_elements]*ms
    seed = lfsr['seed'][prev_index:prev_index+n_elements]*ms
    prev_index += n_elements

    return max_value, init, seed, prev_index

def create_lfsr(neuron_groups, synapse_groups, time_step):
    lfsr_lengths = []
    groups = neuron_groups+synapse_groups
    for g in groups:
        if isinstance(g, Neurons):
            lfsr_lengths.extend(list(g.lfsr_num_bits))
        elif isinstance(g, Connections):
            lfsr_lengths.extend(list(g.lfsr_num_bits_syn))
            if hasattr(g, 'decay_probability_Apre'):
                lfsr_lengths.extend(list(g.lfsr_num_bits_Apre))
                lfsr_lengths.extend(list(g.lfsr_num_bits_Apost))
            if hasattr(g, 'counter_Apre'):
                lfsr_lengths.extend(list(g.lfsr_num_bits_condApre1))
                lfsr_lengths.extend(list(g.lfsr_num_bits_condApre2))
                lfsr_lengths.extend(list(g.lfsr_num_bits_condApost1))
                lfsr_lengths.extend(list(g.lfsr_num_bits_condApost2))
    lfsr_num_bits = np.unique(lfsr_lengths)
    lfsr_max = [int(2**n - 1) for n in lfsr_num_bits]

    mask = {3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011,
            9: 0b1000010001}

    lfsr = {}
    lfsr['array'] = []
    init_map = {}
    prev_index = 0
    prev_nbits = 0

    for i, n_bits in enumerate(lfsr_num_bits):
        # Determines possition of lfsr['array'] where each lfsr starts
        init_map[n_bits] = prev_index + (2**prev_nbits - 1)
        prev_index = init_map[n_bits]
        prev_nbits = n_bits

        # Create LFSR values
        lfsr_val = 1
        lfsr['array'].append(lfsr_val)
        for _ in range(1, lfsr_max[i]):
            lfsr_val = lfsr_val << 1
            overflow = lfsr_val >> int(n_bits)
            if overflow:
                lfsr_val ^= mask[n_bits]
            lfsr['array'].append(lfsr_val)
    lfsr_timedarray = TimedArray(lfsr['array'], dt=time_step)

    # Creates values for all element following the order of their
    # initializations
    lfsr['seed'] = [np.random.randint(2**x-1) for x in lfsr_lengths]
    lfsr['max_value'] = [2**x-1 for x in lfsr_lengths]
    lfsr['init'] = [init_map[x] for x in lfsr_lengths]

    prev_index = 0
    for g in (groups):
        if isinstance(g, Neurons):
            g.lfsr_max_value, g.lfsr_init, g.seed, prev_index = (
                    get_parameters(len(g.lfsr_num_bits), lfsr, prev_index)
            )
            # refrac_tau is usually much smaller than tau, so only latter
            # is tested
            if np.any(g.tau >= g.lfsr_max_value):
                raise ValueError(f'Time constant of group {g} too high for\
                                   LFSR length.')
        elif isinstance(g, Connections):
            g.lfsr_max_value_syn, g.lfsr_init_syn, g.seed_syn, prev_index = (
                    get_parameters(len(g.lfsr_num_bits_syn), lfsr, prev_index)
            )
            if np.any(g.tausyn >= g.lfsr_max_value_syn):
                raise ValueError(f'Time constant of group {g} too high for\
                                   LFSR length.')
            if hasattr(g, 'decay_probability_Apre'):
                g.lfsr_max_value_Apre, g.lfsr_init_Apre, g.seed_Apre, prev_index = (
                        get_parameters(len(g.lfsr_num_bits_Apre), lfsr, prev_index)
                )
                if np.any(g.taupre >= g.lfsr_max_value_Apre):
                    raise ValueError(f'Time constant of group {g} too high\
                                       for LFSR length.')
                g.lfsr_max_value_Apost, g.lfsr_init_Apost, g.seed_Apost, prev_index = (
                        get_parameters(len(g.lfsr_num_bits_Apost), lfsr, prev_index)
                )
                if np.any(g.taupost >= g.lfsr_max_value_Apost):
                    raise ValueError(f'Time constant of group {g} too high\
                                       for LFSR length.')
            if hasattr(g, 'counter_Apre'):
                g.lfsr_max_value_condApre1, g.lfsr_init_condApre1, g.seed_condApre1, prev_index = (
                        get_parameters(len(g.lfsr_num_bits_condApre1), lfsr, prev_index)
                )
                g.lfsr_max_value_condApre2, g.lfsr_init_condApre2, g.seed_condApre2, prev_index = (
                        get_parameters(len(g.lfsr_num_bits_condApre2), lfsr, prev_index)
                )
                g.lfsr_max_value_condApost1, g.lfsr_init_condApost1, g.seed_condApost1, prev_index = (
                        get_parameters(len(g.lfsr_num_bits_condApost1), lfsr, prev_index)
                )
                g.lfsr_max_value_condApost2, g.lfsr_init_condApost2, g.seed_condApost2, prev_index = (
                        get_parameters(len(g.lfsr_num_bits_condApost2), lfsr, prev_index)
                )

        g.namespace['lfsr_timedarray'] = lfsr_timedarray

    return lfsr_timedarray
