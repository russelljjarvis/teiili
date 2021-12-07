"""
Created on Mon May 03 2021

@author=pabloabur

This file contains main parameters and definitions of the building block
related to quantized stochastic models described by Wang et al. (2018).
The dictionaries provided below represent motifs for connections and groups.
The descriptor will filter values according to layer and plasticity rules, so
that scaling up is more modular and (hopefully) more organized. Therefore,
global dictionaries below represent motifs that will be selected according to
laminar and plasticity options provided. Parameters that require fine
tunning can be used with filter_params methods of classes below.
Note that we chose to be explicit, so there are a lot of variables. The final
descriptor however contains only the desired parameters from motifs.
"""
import os
import copy

from brian2 import ms, mV, mA
import numpy as np

from teili.models.neuron_models import QuantStochLIF as static_neuron_model
from teili.models.synapse_models import QuantStochSyn as static_synapse_model
from teili.models.synapse_models import QuantStochSynStdp as stdp_synapse_model
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

# Dictionaries used as Lookup table to construct descriptor
syn_input_prob = {
    'ff_pyr': 0.7,
    'ff_pv': 1.0,
    'ff_sst': 1.0,
    'ff_vip': 0.0,
    'fb_pyr': 1.0,
    'fb_pv': 0.0,
    'fb_sst': 0.0,
    'fb_vip': 1.0
    }

syn_input_plast = {
    'ff_pyr': 'reinit',
    'ff_pv': 'static',
    'ff_sst': 'static',
    'ff_vip': 'static',
    #'fb_pyr': 'stdp',
    #'fb_vip': 'stdp'
    }

syn_intra_prob = {
    'L23': {
        'pyr_pyr': 0.5,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65}, #TODO
    'L4': {
        'pyr_pyr': 0.5, # TODO check. this has been standard. Commented is H&S
        'pyr_pv': 0.15, #TODO 0.45
        'pyr_sst': 0.15, #TODO 0.35
        'pyr_vip': 0.10, #TODO 0.10
        'pv_pyr': 1.0, #TODO 0.60
        'pv_pv': 1.0, #TODO 0.50
        'sst_pv': 0.9, #TODO 0.60
        'sst_pyr': 1.0, #TODO 0.55
        'sst_vip': 0.9, #TODO 0.45
        'vip_sst': 0.65}, #TODO 0.50
    'L5': {
        'pyr_pyr': 0.2,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65}, #TODO
    'L6': {
        'pyr_pyr': 0.2,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65}, #TODO
    }

syn_intra_plast = {
    'L23': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'istdp',
        'pv_pv': 'static',
        'sst_pv': 'altadp',
        'sst_pyr': 'istdp',
        'sst_vip': 'static',
        'vip_sst': 'static'},
    'L4': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'istdp',
        'pv_pv': 'static',
        'sst_pv': 'altadp',
        'sst_pyr': 'istdp',
        'sst_vip': 'static',
        'vip_sst': 'static'},
    'L5': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'istdp',
        'pv_pv': 'static',
        'sst_pv': 'altadp',
        'sst_pyr': 'istdp',
        'sst_vip': 'static',
        'vip_sst': 'static'},
    'L6': {
        'pyr_pyr': 'stdp',
        'pyr_pv': 'static',
        'pyr_sst': 'static',
        'pyr_vip': 'static',
        'pv_pyr': 'istdp',
        'pv_pv': 'static',
        'sst_pv': 'altadp',
        'sst_pyr': 'istdp',
        'sst_vip': 'static',
        'vip_sst': 'static'},
    }

interlaminar_conn_prob = {
    'L23_L4': {
        'pyr_pyr': 0.03,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65}, #TODO
    'L23_L5': {
        'pyr_pyr': 0.04,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65}, #TODO
    'L23_L6': {
        'pyr_pyr': 0.03,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65}, #TODO
    'L4_L23': {
        'pyr_pyr': 0.05,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65}, #TODO
    'L4_L5': {
        'pyr_pyr': 0.01,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65}, #TODO
    'L4_L6': {
        'pyr_pyr': 0.02,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65}, #TODO
    'L5_L23': {
        'pyr_pyr': 0.03,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65}, #TODO
    'L5_L4': {
        'pyr_pyr': 0.0,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65}, #TODO
    'L5_L6': {
        'pyr_pyr': 0.01,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65}, #TODO
    'L6_L4': {
        'pyr_pyr': 0.1,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65}, #TODO
    'L6_L23': {
        'pyr_pyr': 0.0,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65}, #TODO
    'L6_L5': {
        'pyr_pyr': 0.0,
        'pyr_pv': 0.15, #TODO
        'pyr_sst': 0.15, #TODO
        'pyr_vip': 0.10, #TODO
        'pv_pyr': 1.0, #TODO
        'pv_pv': 1.0, #TODO
        'sst_pv': 0.9, #TODO
        'sst_pyr': 1.0, #TODO
        'sst_vip': 0.9, #TODO
        'vip_sst': 0.65} #TODO
    }

# The dictionaries below contain the parameters for each case, as defined above
syn_base_vals = {
    'static': {
        'ff_pyr': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'ff_pv': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'ff_sst': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'ff_vip': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'pyr_pyr': {
            'gain_syn': 1*mA,
            'weight': 1,
            'w_plast': 1,
            'delay': 4*ms},
        'pyr_pv': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'pyr_sst': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'pyr_vip': {
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'pv_pyr': {
            'gain_syn': 1*mA,
            'weight': -2,
            'w_plast': 1,
            'delay': 0*ms},
        'pv_pv': {
            'gain_syn': 1*mA,
            'weight': -1,
            'w_plast': 1,
            'delay': 0*ms},
        'sst_pyr': {
            'gain_syn': 1*mA,
            'weight': -2,
            'w_plast': 1,
            'delay': 0*ms},
        'sst_pv': {
            'gain_syn': 1*mA,
            'weight': -2,
            'w_plast': 1,
            'delay': 0*ms},
        'sst_vip': {
            'gain_syn': 1*mA,
            'weight': -1,
            'w_plast': 1,
            'delay': 0*ms},
        'vip_sst': {
            'gain_syn': 1*mA,
            'weight': -1,
            'w_plast': 1,
            'gain_syn': 1*mA,
            'delay': 0*ms},
        },
    'reinit': {
        'ff_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 2,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'stdp_thres': 1}
        },
    'istdp': {
        'pv_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'weight': -1,
            'w_plast': 1,
            'w_max': lambda n_bits: 2**(n_bits - 1),
            'stdp_thres': 1},
        'sst_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'weight': -1,
            'w_plast': 1,
            'w_max': lambda n_bits: 2**(n_bits - 1),
            'stdp_thres': 1},
        },
    'adp': {
        'pv_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'weight': -1,
            'w_plast': 1,
            'w_max': lambda n_bits: 2**(n_bits - 1),
            'variance_th': 0.50,
            'stdp_thres': 1},
        'sst_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'rand_num_bits': lambda n_bits: n_bits,
            'weight': -1,
            'w_plast': 1,
            'w_max': lambda n_bits: 2**(n_bits - 1),
            'variance_th': 0.50,
            'stdp_thres': 1},
        },
    'altadp': {
        'sst_pv': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'weight': -1,
            'w_plast': 1,
            'w_max': lambda n_bits: 2**(n_bits - 1),
            'inh_learning_rate': 0.01,
            'stdp_thres': 1
            }
        },
    'stdp': {
        'ff_pyr': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 2,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'stdp_thres': 1
            },
        'ff_pv': {
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 2,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'stdp_thres': 1
            },
        'pyr_pyr': {
            'gain_syn': 1*mA,
            'delay': 4*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'A_max': lambda n_bits: 2**n_bits - 1,
            'rand_num_bits': lambda n_bits: n_bits,
            'weight': 1,
            'w_plast': 4,
            'w_max': lambda n_bits: 2**(n_bits - 1) - 1,
            'stdp_thres': 1
            },
        }
    }

# Values of type string must correspond to a key in base_vals (except variable)
syn_sample_vars = {
    'static': {
        'ff_pyr': [
            {'variable': 'weight', 'unit': 1,
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            ],
        'ff_pv': [
            {'variable': 'weight', 'unit': 1,
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            ],
        'ff_sst': [
            {'variable': 'weight', 'unit': 1,
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            ],
        'ff_vip': [
            {'variable': 'weight', 'unit': 1,
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            ],
        'pyr_pyr': [
            {'variable': 'weight', 'unit': 1,
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            {'variable': 'delay', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': 8},
            ],
        'pyr_pv': [
            {'variable': 'weight', 'unit': 1,
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            ],
        'pyr_sst': [
            {'variable': 'weight', 'unit': 1,
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            ],
        'pyr_vip': [
            {'variable': 'weight', 'unit': 1,
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1) - 1},
            ],
        'pv_pyr': [
            {'variable': 'weight', 'unit': 1,
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1)},
            ],
        'pv_pv': [
            {'variable': 'weight', 'unit': 1,
                'sign': lambda weight: np.sign(weight), 'min': 1,
                'max': lambda n_bits: 2**(n_bits - 1)},
            ],
        'sst_pv': [
                {'variable': 'weight', 'unit': 1,
                    'sign': lambda weight: np.sign(weight), 'min': 1,
                    'max': lambda n_bits: 2**(n_bits - 1)},
            ],
        'sst_pyr': [
                {'variable': 'weight', 'unit': 1,
                    'sign': lambda weight: np.sign(weight), 'min': 1,
                    'max': lambda n_bits: 2**(n_bits - 1)},
            ],
        'sst_vip': [
                {'variable': 'weight', 'unit': 1,
                    'sign': lambda weight: np.sign(weight), 'min': 1,
                    'max': lambda n_bits: 2**(n_bits - 1)},
            ],
        'vip_sst': [
                {'variable': 'weight', 'unit': 1,
                    'sign': lambda weight: np.sign(weight), 'min': 1,
                    'max': lambda n_bits: 2**(n_bits - 1)},
            ],
        },
    'reinit': {
        'ff_pyr': [
            {'variable': 'w_plast', 'unit': 1, 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            {'variable': 'taupre', 'unit': 1*ms, 'sign': 1, 'min': 0,
                'max': lambda max_tau: max_tau},
            {'variable': 'taupost', 'unit': 1*ms, 'sign': 1, 'min': 0,
                'max': lambda max_tau: max_tau},
            ]
        },
    'istdp': {
        'pv_pyr': [
            {'variable': 'w_plast', 'unit': 1, 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            {'variable': 'taupre', 'unit': 1*ms, 'sign': 1, 'min': 0,
                'max': lambda max_tau: max_tau},
            {'variable': 'taupost', 'unit': 1*ms, 'sign': 1, 'min': 0,
                'max': lambda max_tau: max_tau},
            ],
        'sst_pyr': [
            {'variable': 'w_plast', 'unit': 1, 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            {'variable': 'taupre', 'unit': 1*ms, 'sign': 1, 'min': 0,
                'max': lambda max_tau: max_tau},
            {'variable': 'taupost', 'unit': 1*ms, 'sign': 1, 'min': 0,
                'max': lambda max_tau: max_tau},
            ]
        },
    'altadp': {
        'sst_pv': [
            {'variable': 'w_plast', 'unit': 1, 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            ]
        },
    'stdp': {
        'ff_pyr': [
            {'variable': 'w_plast', 'unit': 1, 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            {'variable': 'taupre', 'unit': 1*ms, 'sign': 1, 'min': 0,
                'max': lambda max_tau: max_tau},
            {'variable': 'taupost', 'unit': 1*ms, 'sign': 1, 'min': 0,
                'max': lambda max_tau: max_tau},
            ],
        'ff_pv': [
            {'variable': 'w_plast', 'unit': 1, 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            {'variable': 'taupre', 'unit': 1*ms, 'sign': 1, 'min': 0,
                'max': lambda max_tau: max_tau},
            {'variable': 'taupost', 'unit': 1*ms, 'sign': 1, 'min': 0,
                'max': lambda max_tau: max_tau},
            ],
        'pyr_pyr': [
            {'variable': 'w_plast', 'unit': 1, 'sign': 1, 'min': 1,
                'max': lambda w_max: w_max},
            {'variable': 'delay', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': 8},
            {'variable': 'taupre', 'unit': 1*ms, 'sign': 1, 'min': 0,
                'max': lambda max_tau: max_tau},
            {'variable': 'taupost', 'unit': 1*ms, 'sign': 1, 'min': 0,
                'max': lambda max_tau: max_tau},
            ]
        }
    }

reinit_vars = {
    'ff_pyr': {
        're_init_dt': 60000*ms
        }
    }

# Dictionaries of neuronal populations
neu_pop = {
    'L23': {
        'n_exc': 57,
        'ei_ratio': 4,
        'inh_ratio': {
            'pv_cells': .37,
            'sst_cells': .20,
            'vip_cells': .43
            },
        'num_inputs': {
            'pyr_cells': 4,
            'pv_cells': 4,
            'sst_cells': 3,
            'vip_cells': 2
            },
        },
    'L4': {
        'n_exc': 49,
        'ei_ratio': 4, # TODO 1.6 in case I need more inhibition?
        # TODO use previous values if I want only PVs
        'inh_ratio': {
            'pv_cells': .68,#1
            'sst_cells': .20,#.02,
            'vip_cells': .12,#.02
            },
        'num_inputs': {
            'pyr_cells': 4,
            'pv_cells': 4,
            'sst_cells': 3,
            'vip_cells': 2
            },
        },
    'L5': {
        'n_exc': 53,
        'ei_ratio': 6,
        'inh_ratio': {
            'pv_cells': .52,
            'sst_cells': .37,
            'vip_cells': .11
            },
        'num_inputs': {
            'pyr_cells': 4,
            'pv_cells': 4,
            'sst_cells': 3,
            'vip_cells': 2
            },
        },
    'L6': {
        'n_exc': 98,
        'ei_ratio': 6,
        'inh_ratio': {
            'pv_cells': .49,
            'sst_cells': .38,
            'vip_cells': .13
            },
        'num_inputs': {
            'pyr_cells': 4,
            'pv_cells': 4,
            'sst_cells': 3,
            'vip_cells': 2
            },
        },
    }

neu_pop_plast = {
    'pyr_cells': 'adapt',
    'pv_cells': 'adapt',
    'sst_cells': 'static',
    'vip_cells': 'static'
    }

neu_base_vals = {
    'static': {
        'pyr_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'Vm_noise': 0*mV
            },
        'pv_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'Vm_noise': 0*mV},
        'sst_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'Vm_noise': 0*mV},
        'vip_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'Vm_noise': 0*mV},
        },
    'adapt': {
        'pyr_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'thr_min': lambda n_bits: np.ceil(2**n_bits/5)*mV,
            'thr_max': lambda n_bits: (2**n_bits - 1)*mV,
            'Vm_noise': 0*mV
            },
        'pv_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'thr_min': lambda n_bits: np.ceil(2**n_bits/5)*mV,
            'thr_max': lambda n_bits: (2**n_bits - 1)*mV,
            'Vm_noise': 0*mV},
        'sst_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'thr_min': lambda n_bits: np.ceil(2**n_bits/5)*mV,
            'thr_max': lambda n_bits: (2**n_bits - 1)*mV,
            'Vm_noise': 0*mV},
        'vip_cells': {
            'tau': 20*ms,
            'decay_numerator': 244,
            'refrac_tau': 2*ms,
            'refrac_decay_numerator': 154,
            'tausyn': 5*ms,
            'syn_decay_numerator': 213,
            'rand_num_bits': lambda n_bits: n_bits,
            'Vm': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vrest': lambda n_bits: (2**n_bits - 1)/5*mV,
            'Vm_max': lambda n_bits: (2**n_bits - 1),
            'Vthr': lambda n_bits: (2**n_bits - 1)*mV,
            'I_min': lambda n_bits: -2**(n_bits-1)*mA,
            'I_max': lambda n_bits: (2**(n_bits-1) - 1)*mA,
            'thr_min': lambda n_bits: np.ceil(2**n_bits/5)*mV,
            'thr_max': lambda n_bits: (2**n_bits - 1)*mV,
            'Vm_noise': 0*mV}
        },
    }

neu_sample_vars = {
    'pyr_cells': [
        {'variable': 'decay_numerator', 'unit': 1, 'sign': 1, 'min': 1, 'max': 255},
        {'variable': 'syn_decay_numerator', 'unit': 1, 'sign': 1, 'min': 1, 'max': 255},
        ],
    'pv_cells': [
        {'variable': 'decay_numerator', 'unit': 1, 'sign': 1, 'min': 1, 'max': 255},
        {'variable': 'syn_decay_numerator', 'unit': 1, 'sign': 1, 'min': 1, 'max': 255},
        ],
    'sst_cells': [
        {'variable': 'decay_numerator', 'unit': 1, 'sign': 1, 'min': 1, 'max': 255},
        {'variable': 'syn_decay_numerator', 'unit': 1, 'sign': 1, 'min': 1, 'max': 255},
        ],
    'vip_cells': [
        {'variable': 'decay_numerator', 'unit': 1, 'sign': 1, 'min': 1, 'max': 255},
        {'variable': 'syn_decay_numerator', 'unit': 1, 'sign': 1, 'min': 1, 'max': 255},
        ],
    }

class ParameterDescriptor:
    """ Parent class that contains parameters.
    Attributes:
        layer (str): Represent layer.
        model_path (str): Path to where models are stored.
        constants (dict): Constants that will be used for all elements.
        models (dict): Models of the class.
    """
    def __init__(self, layer, path):
        self.layer = layer
        self.constants = {'n_bits': 4, 'max_tau': 2**5 - 1}
        path = os.path.expanduser(path)
        self.model_path = os.path.join(path, "teili", "models", "equations", "")
        self.models = {}

class ConnectionDescriptor(ParameterDescriptor):
    """ This class describes the standard characterists of the connections. 

    Attributes:
        models (dict): Synaptic models available.
        input_prob (dict): Connection probability of inputs.
        input_plast (dict): Plasticity types of inputs.
        intra_prob (dict): Connection probability of intralaminar projections.
        intra_plast (dict): Plasticity types of intralaminar projections.
        base_vals (dict): General paramaters, as defined by layer and plasticty
            types defined.
        reinit_var (dict): Variables used for plasticity of type 'reinit'.
    """
    def __init__(self, layer, path):
        super().__init__(layer, path)
        self.models['static'] = SynapseEquationBuilder(base_unit='quantized',
            plasticity='non_plastic')
        self.models['stdp'] = SynapseEquationBuilder(base_unit='quantized',
            plasticity='quantized_stochastic_stdp')
        self.models['adp'] = SynapseEquationBuilder.import_eq(
                self.model_path + 'StochSynAdp.py')
        self.models['altadp'] = SynapseEquationBuilder.import_eq(
                self.model_path + 'StochAdpIin.py')
        self.models['istdp'] = SynapseEquationBuilder.import_eq(
                self.model_path + 'StochInhStdp.py')
        self.models['reinit'] = SynapseEquationBuilder(base_unit='quantized',
                plasticity='quantized_stochastic_stdp',
                structural_plasticity='stochastic_counter')

        self.input_prob = syn_input_prob
        self.input_plast = syn_input_plast
        self.intra_prob = syn_intra_prob[self.layer]
        self.intra_plast = syn_intra_plast[self.layer]
        self.base_vals = {}
        self.sample = {}
        self.reinit_vars = {}

    def filter_params(self):
        """ Update parameters that will be used to build synapse model. This
            is done by changing the values of attributes base_vals, sample, 
            and reinit_vars according to what was set in intra_plas,
            input_plast, and constants.
        """
        conn_groups = [self.intra_plast, self.input_plast]
        for conn_group in conn_groups:
            for conn, plast in conn_group.items():
                self.base_vals[conn] = process_base_vars(
                    syn_base_vals[plast][conn],
                    self.constants)

        for conn_group in conn_groups:
            for conn, plast in conn_group.items():
                self.sample[conn] = process_sample_vars(
                    syn_sample_vars[plast][conn],
                    {**self.base_vals[conn], **self.constants})

        for conn, plast in self.input_plast.items():
            # At the moment only works for reinit connections
            if plast == 'reinit':
                self.reinit_vars[conn] = reinit_vars[conn]

class PopulationDescriptor(ParameterDescriptor):
    """ This class describes the standard characterists of the populations.

    Attributes:
        models (dict): Neuronal models available.
        pop (dict): Contains general information about the population.
        base_vals (dict): General parameters, as defined by layer.
    """
    def __init__(self, layer, path):
        super().__init__(layer, path)
        self.models['static'] = NeuronEquationBuilder(base_unit='quantized',
            position='spatial')
        self.models['adapt'] = NeuronEquationBuilder(base_unit='quantized',
            intrinsic_excitability='threshold_adaptation',
            position='spatial')

        self.group_vals = neu_pop[self.layer]
        self.group_plast = neu_pop_plast
        self.base_vals = {}
        self.sample = {}
        self.groups = {}
        
    def filter_params(self):
        """ Filter parameters that will be used to build neuron model. This
            is done by changing the values of attributes base_vals, and sample
            according to what was set in group_vals, group_plast, and constants.
        """
        temp_pop = {}
        num_inh = int(self.group_vals['n_exc']/self.group_vals['ei_ratio'])
        temp_pop['pyr_cells'] = {'num_neu': self.group_vals['n_exc']}
        for inh_pop, ratio in self.group_vals['inh_ratio'].items():
            temp_pop[inh_pop] = {'num_neu': int(num_inh * ratio)}
            #num_pv = num_pv if num_pv else 1
        for pop, n_inp in self.group_vals['num_inputs'].items():
            temp_pop[pop].update({'num_inputs': n_inp})
        self.groups = temp_pop
        for conn, plast in self.group_plast.items():
            self.base_vals[conn] = process_base_vars(
                neu_base_vals[plast][conn],
                self.constants)

        for neu_group in self.group_plast.keys():
            self.sample[neu_group] = process_sample_vars(
                neu_sample_vars[neu_group],
                {**self.base_vals[neu_group], **self.constants})

def process_base_vars(base_objects, reference_vals):
    ''' This function filter the necessary parameters from provided
        reference dictionaries to determine base values.

    Args:
        base_objects (dict of list): Contains keys and values
            identifying the parameters. If it contains a
            lambda function, its argument name must be present
            as a key in reference_vals.
        reference_vals (dict): Base parameters used on base
            values.
    Returns:
        values_list (list): Contains parameters that will be used
            as base paramaters.
    '''
    processed_objects = copy.deepcopy(base_objects)
    for var in base_objects:
        if callable(base_objects[var]):
            processed_objects[var] = process_dynamic_values(
                base_objects[var], reference_vals)

    return processed_objects

def process_sample_vars(sample_objects, reference_vals):
    ''' This function filter the necessary parameters from provided
        reference dictionaries to determine sampling process.

    Args:
        sample_objects (dict of list): Contains keys and values
            identifying the samplying process. If it contains a
            lambda function, its argument name must be present
            as a key in reference_vals.
        reference_vals (dict): Base parameters used on sampling
            process.
    Returns:
        sample_list (list): Contains parameters that will be used
            by sampling function.
    '''
    sample_list = []
    for sample_var in sample_objects:
        # Get values from base val if they are not numbers
        variable = sample_var['variable']
        variable_mean = reference_vals[variable]
        unit = sample_var['unit']
        sign = sample_var['sign']
        if callable(sign):
            sign = process_dynamic_values(sign, reference_vals)
        unit *= sign
        variable_mean /= np.abs(unit)
        clip_min = sample_var['min']
        if callable(clip_min):
            clip_min = process_dynamic_values(clip_min, reference_vals)
        clip_max = sample_var['max']
        if callable(clip_max):
            clip_max = process_dynamic_values(clip_max, reference_vals)

        sample_list.append({'variable': variable,
                             'dist_param': np.abs(variable_mean),
                             'unit': unit,
                             'clip_min': clip_min,
                             'clip_max': clip_max})

    return sample_list

def process_dynamic_values(lambda_func, reference_dict):
    """ Evaluates parameters defined as lambda functions.

    Args:
        lambda_func (callable): Function to be evaluated.
        reference_dict (dict): Contains function's argument
            as a key.
    """
    var_name = lambda_func.__code__.co_varnames[0]
    return lambda_func(reference_dict[var_name])

layer = 'L4'
path = '/Users/Pablo/git/teili/'
conn_desc = ConnectionDescriptor(layer, path)
pop_desc = PopulationDescriptor(layer, path)
pop_desc.filter_params()
conn_desc.filter_params()
