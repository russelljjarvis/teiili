"""
Created on Mon May 03 2021

@author=pabloabur

This file contains parameters of the building block related to quantized
stochastic models described by Wang et al. (2018). The dictionaries provided
below are the usual values for all connections and groups used. The final
descriptor will filter values according to layer and plasticity rules, so
that scaling up is more modular and (hopefully) more organized. Therefore,
change dictionaries below if you want a different motif and let the class
build the descriptor. Note that we chose to be explicit, so there is a lot
of variables.

"""
import os

from brian2 import ms, mV, mA
import numpy as np

from teili.models.neuron_models import QuantStochLIF as static_neuron_model
from teili.models.synapse_models import QuantStochSyn as static_synapse_model
from teili.models.synapse_models import QuantStochSynStdp as stdp_synapse_model
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

MAX_WEIGHT = 2**4 - 1
MAX_TAU = 2**5 - 1

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
            'tausyn': 5*ms,
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'ff_pv': {
            'tausyn': 5*ms,
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'ff_sst': {
            'tausyn': 5*ms,
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'ff_vip': {
            'tausyn': 5*ms,
            'gain_syn': 1*mA,
            'weight': 3,
            'w_plast': 1,
            'delay': 0*ms},
        'pyr_pyr': {
            'tausyn': 5*ms,
            'gain_syn': 1*mA,
            'weight': 1,
            'w_plast': 1,
            'delay': 4*ms},
        'pyr_pv': {
            'tausyn': 5*ms,
            'gain_syn': 1*mA,
            'weight': 3,# 1,#TODO sharp test
            'w_plast': 1,
            'delay': 0*ms},
        'pyr_sst': {
            'tausyn': 5*ms,
            'gain_syn': 1*mA,
            'weight': 3,# 1,#TODO sharp test
            'w_plast': 1,
            'delay': 0*ms},
        'pyr_vip': {
            'tausyn': 5*ms,
            'gain_syn': 1*mA,
            'weight': 3,# 1,#TODO sharp test
            'w_plast': 1,
            'delay': 0*ms},
        'pv_pyr': {
            'tausyn': 10*ms,
            'gain_syn': 1*mA,
            'weight': -2,
            'w_plast': 1,
            'delay': 0*ms},
        'pv_pv': {
            'tausyn': 10*ms,
            'gain_syn': 1*mA,
            'weight': -1,
            'w_plast': 1,
            'delay': 0*ms},
        'sst_pyr': {
            'tausyn': 10*ms,
            'gain_syn': 1*mA,
            'weight': -2,
            'w_plast': 1,
            'delay': 0*ms},
        'sst_pv': {
            'tausyn': 10*ms,
            'gain_syn': 1*mA,
            'weight': -2,
            'w_plast': 1,
            'delay': 0*ms},
        'sst_vip': {
            'tausyn': 10*ms,
            'gain_syn': 1*mA,
            'weight': -1,
            'w_plast': 1,
            'delay': 0*ms},
        'vip_sst': {
            'tausyn': 10*ms,
            'weight': -1,
            'w_plast': 1,
            'gain_syn': 1*mA,
            'delay': 0*ms},
        },
    'reinit': {
        'ff_pyr': {
            'tausyn': 5*ms,
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'rand_num_bits_Apre': 4,
            'rand_num_bits_Apost': 4,
            'weight': 1,
            'w_plast': 2,
            'w_max': MAX_WEIGHT,
            'stdp_thres': 1}
        },
    'istdp': {
        'pv_pyr': {
            'tausyn': 10*ms,
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'rand_num_bits_Apre': 4,
            'rand_num_bits_Apost': 4,
            'weight': -1,
            'w_plast': 1,
            'w_max': 15,
            'stdp_thres': 1},
        'sst_pyr': {
            'tausyn': 10*ms,
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'rand_num_bits_Apre': 4,
            'rand_num_bits_Apost': 4,
            'weight': -1,
            'w_plast': 1,
            'w_max': 15,
            'stdp_thres': 1},
        },
    'adp': {
        'pv_pyr': {
            'tausyn': 10*ms,
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'rand_num_bits_Apre': 4,
            'rand_num_bits_Apost': 4,
            'weight': -1,
            'w_plast': 1,
            'w_max': MAX_WEIGHT,
            'variance_th': 0.50,
            'stdp_thres': 1},
        'sst_pyr': {
            'tausyn': 10*ms,
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'rand_num_bits_Apre': 4,
            'rand_num_bits_Apost': 4,
            'weight': -1,
            'w_plast': 1,
            'w_max': MAX_WEIGHT,
            'variance_th': 0.50,
            'stdp_thres': 1},
        },
    'altadp': {
        'sst_pv': {
            'tausyn': 5*ms,
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'rand_num_bits_Apre': 4,
            'rand_num_bits_Apost': 4,
            'weight': -1,
            'w_plast': 1,
            'w_max': MAX_WEIGHT,
            'inh_learning_rate': 0.01,
            'stdp_thres': 1
            }
        },
    'stdp': {
        'ff_pyr': {
            'tausyn': 5*ms,
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'rand_num_bits_Apre': 4,
            'rand_num_bits_Apost': 4,
            'weight': 1,
            'w_plast': 2,
            'w_max': MAX_WEIGHT,
            'stdp_thres': 1
            },
        'ff_pv': {
            'tausyn': 5*ms,
            'gain_syn': 1*mA,
            'delay': 0*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'rand_num_bits_Apre': 4,
            'rand_num_bits_Apost': 4,
            'weight': 1,
            'w_plast': 2,
            'w_max': MAX_WEIGHT,
            'stdp_thres': 1
            },
        'pyr_pyr': {
            'tausyn': 5*ms,
            'gain_syn': 1*mA,
            'delay': 4*ms,
            'taupre': 20*ms,
            'taupost': 30*ms,
            'rand_num_bits_Apre': 4,
            'rand_num_bits_Apost': 4,
            'weight': 1,
            'w_plast': 4,
            'w_max': MAX_WEIGHT,
            'stdp_thres': 1
            },
        }
    }

# Values of type string must correspond to a key in base_vals (except variable)
syn_sample_vars = {
    'static': {
        'ff_pyr': [
            {'variable': 'weight', 'unit': 1, 'sign': 'weight', 'min': 1, 'max': MAX_TAU},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'ff_pv': [
            {'variable': 'weight', 'unit': 1, 'sign': 'weight', 'min': 1, 'max': MAX_TAU},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'ff_sst': [
            {'variable': 'weight', 'unit': 1, 'sign': 'weight', 'min': 1, 'max': MAX_TAU},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'ff_vip': [
            {'variable': 'weight', 'unit': 1, 'sign': 'weight', 'min': 1, 'max': MAX_TAU},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'pyr_pyr': [
            {'variable': 'weight', 'unit': 1, 'sign': 'weight', 'min': 1, 'max': MAX_TAU},
            {'variable': 'delay', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': 8},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'pyr_pv': [
            {'variable': 'weight', 'unit': 1, 'sign': 'weight', 'min': 1, 'max': MAX_TAU},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'pyr_sst': [
            {'variable': 'weight', 'unit': 1, 'sign': 'weight', 'min': 1, 'max': MAX_TAU},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'pyr_vip': [
            {'variable': 'weight', 'unit': 1, 'sign': 'weight', 'min': 1, 'max': MAX_TAU},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'pv_pyr': [
            {'variable': 'weight', 'unit': 1, 'sign': 'weight', 'min': 1, 'max': MAX_TAU},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'pv_pv': [
            {'variable': 'weight', 'unit': 1, 'sign': 'weight', 'min': 1, 'max': MAX_TAU},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'sst_pv': [
            {'variable': 'weight', 'unit': 1, 'sign': 'weight', 'min': 1, 'max': MAX_TAU},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'sst_pyr': [
            {'variable': 'weight', 'unit': 1, 'sign': 'weight', 'min': 1, 'max': MAX_TAU},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'sst_vip': [
            {'variable': 'weight', 'unit': 1, 'sign': 'weight', 'min': 1, 'max': MAX_TAU},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'vip_sst': [
            {'variable': 'weight', 'unit': 1, 'sign': 'weight', 'min': 1, 'max': MAX_TAU},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        },
    'reinit': {
        'ff_pyr': [
            {'variable': 'w_plast', 'unit': 1, 'sign': 1, 'min': 1, 'max': 'w_max'},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            {'variable': 'taupre', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            {'variable': 'taupost', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ]
        },
    'istdp': {
        'pv_pyr': [
            {'variable': 'w_plast', 'unit': 1, 'sign': 1, 'min': 1, 'max': 'w_max'},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            {'variable': 'taupre', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            {'variable': 'taupost', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'sst_pyr': [
            {'variable': 'w_plast', 'unit': 1, 'sign': 1, 'min': 1, 'max': 'w_max'},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            {'variable': 'taupre', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            {'variable': 'taupost', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ]
        },
    'altadp': {
        'sst_pv': [
            {'variable': 'w_plast', 'unit': 1, 'sign': 1, 'min': 1, 'max': 'w_max'},
            ]
        },
    'stdp': {
        'ff_pyr': [
            {'variable': 'w_plast', 'unit': 1, 'sign': 1, 'min': 1, 'max': 'w_max'},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            {'variable': 'taupre', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            {'variable': 'taupost', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'ff_pv': [
            {'variable': 'w_plast', 'unit': 1, 'sign': 1, 'min': 1, 'max': 'w_max'},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            {'variable': 'taupre', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            {'variable': 'taupost', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            ],
        'pyr_pyr': [
            {'variable': 'w_plast', 'unit': 1, 'sign': 1, 'min': 1, 'max': 'w_max'},
            {'variable': 'delay', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': 8},
            {'variable': 'tausyn', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            {'variable': 'taupre', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
            {'variable': 'taupost', 'unit': 1*ms, 'sign': 1, 'min': 0, 'max': MAX_TAU},
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
            'Vm': 3*mV,
            'Vm_noise': 0*mV
            },
        'pv_cells': {
            'tau': 20*ms,
            'Vm': 3*mV,
            'Vm_noise': 0*mV},
        'sst_cells': {
            'tau': 20*ms,
            'Vm': 3*mV,
            'Vm_noise': 0*mV},
        'vip_cells': {
            'tau': 20*ms,
            'Vm': 3*mV,
            'Vm_noise': 0*mV},
        },
    'adapt': {
        'pyr_cells': {
            'tau': 20*ms,
            'Vm': 3*mV,
            'thr_max': 15*mV,
            'Vm_noise': 0*mV
            },
        'pv_cells': {
            'tau': 20*ms,
            'Vm': 3*mV,
            'thr_max': 15*mV,
            'Vm_noise': 0*mV},
        'sst_cells': {
            'tau': 20*ms,
            'Vm': 3*mV,
            'Vm_noise': 0*mV},
        'vip_cells': {
            'tau': 20*ms,
            'Vm': 3*mV,
            'Vm_noise': 0*mV}
        },
    }

neu_sample_vars = {
    'pyr_cells': [
        {'variable': 'tau', 'unit': 1*ms, 'sign': 1, 'min': 1, 'max': MAX_TAU}
        ],
    'pv_cells': [
        {'variable': 'tau', 'unit': 1*ms, 'sign': 1, 'min': 1, 'max': MAX_TAU}
        ],
    'sst_cells': [
        {'variable': 'tau', 'unit': 1*ms, 'sign': 1, 'min': 1, 'max': MAX_TAU}
        ],
    'vip_cells': [
        {'variable': 'tau', 'unit': 1*ms, 'sign': 1, 'min': 1, 'max': MAX_TAU}
        ],
    }

class ConnectionDescriptor:
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
        path = os.path.expanduser(path)
        model_path = os.path.join(path, "teili", "models", "equations", "")

        self.models = {}
        self.models['stdp'] = stdp_synapse_model
        self.models['static'] = static_synapse_model
        self.models['adp'] = SynapseEquationBuilder.import_eq(
                model_path + 'StochSynAdp.py')
        self.models['altadp'] = SynapseEquationBuilder.import_eq(
                model_path + 'StochAdpIin.py')
        self.models['istdp'] = SynapseEquationBuilder.import_eq(
                model_path + 'StochInhStdp.py')
        self.models['reinit'] = SynapseEquationBuilder(base_unit='quantized',
                plasticity='quantized_stochastic_stdp',
                structural_plasticity='stochastic_counter')

        self.input_prob = syn_input_prob
        self.input_plast = syn_input_plast
        self.intra_prob = syn_intra_prob[layer]
        self.intra_plast = syn_intra_plast[layer]
        self.base_vals = {}
        self.sample_vars = syn_sample_vars
        self.sample = {}
        self.reinit_vars = {}
        self.update_params()

    def update_params(self):
        conn_groups = [self.intra_plast, self.input_plast]
        for conn_group in conn_groups:
            for conn, plast in conn_group.items():
                self.base_vals[conn] = syn_base_vals[plast][conn]

        for conn_group in conn_groups:
            for conn, plast in conn_group.items():
                self.sample[conn] = process_sample_vars(
                    self.sample_vars[plast][conn],
                    self.base_vals[conn])

        for conn, plast in self.input_plast.items():
            if plast == 'reinit':
                self.reinit_vars[conn] = reinit_vars[conn]

class PopulationDescriptor:
    """ This class describes the standard characterists of the populations.

    Attributes:
        models (dict): Neuronal models available.
        pop (dict): Contains general information about the population.
        base_vals (dict): General parameters, as defined by layer.
    """
    def __init__(self, layer, path):
        path = os.path.expanduser(path)
        model_path = os.path.join(path, "teili", "models", "equations", "")

        self.models = {}
        self.models['static'] = static_neuron_model
        self.models['adapt'] = NeuronEquationBuilder(base_unit='quantized',
                intrinsic_excitability='threshold_adaptation',
                position='spatial')

        self.group_vals = neu_pop[layer]
        self.group_plast = neu_pop_plast
        self.base_vals = {}
        self.sample_vars = neu_sample_vars
        self.sample = {}
        self.groups = {}
        self.update_params()
        
    def update_params(self):
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
            self.base_vals[conn] = neu_base_vals[plast][conn]

        for neu_group, sample_objs in self.sample_vars.items():
            self.sample[neu_group] = process_sample_vars(
                sample_objs,
                self.base_vals[neu_group])

def process_sample_vars(sample_objects, reference_vals):
    ''' This function filter the necessary parameters from provided
        dictionaries to determine sampling process.

    Attributes:
        sample_objects (dict of list): Contains keys and values
            identifying the samplying process.
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
        if isinstance(sign, str):
            sign = np.sign(reference_vals[sign])
        unit *= sign
        variable_mean /= np.abs(unit)
        clip_min = sample_var['min']
        if isinstance(clip_min, str):
            clip_min = reference_vals[clip_min]
        clip_max = sample_var['max']
        if isinstance(clip_max, str):
            clip_max = reference_vals[clip_max]

        sample_list.append({'variable': variable,
                             'dist_param': np.abs(variable_mean),
                             'unit': unit,
                             'clip_min': clip_min,
                             'clip_max': clip_max})

    return sample_list

layer = 'L4'
path = '/Users/Pablo/git/teili/'
conn_desc = ConnectionDescriptor(layer, path)
pop_desc = PopulationDescriptor(layer, path)
pop_desc.update_params()
conn_desc.update_params()
