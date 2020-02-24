#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:11:05 2019

@author: matteo, mmilde

This file contains default parameter for OCTA hierarchical BB.
Parameters are stored as dictionaries.
For more information look into
M. B. Milde (2019) Spike-Based Computational Primitives for
Vision-Based Scene Understanding, PhD diss., Institute of
Neuroinformatics, University of Zurch and ETH Zurich,
Zurich Switzerland.

"""

from brian2 import pF, nS, mV, ms, pA, nA

wta_params = {'we_inp_exc': 100,
              'we_exc_inh': 300,  # 55/50
              'wi_inh_exc': -200,  # -300
              'we_exc_exc': 10.0,  # 45
              'sigm': 2,
              'rp_exc': 1 * ms,
              'rp_inh': 1 * ms,
              'wi_inh_inh': -100,
              'ei_connection_probability': 0.5,
              'ie_connection_probability': 0.66,
              'ii_connection_probability': 0.1
              }


octa_params = {'distribution': 'gamma',
               'dist_param_init': 0.5,  # shape for gamma < 0.5
               'scale_init': 1.0,  # sigma for gamma 1.0
               'dist_param_re_init': 0.4,
               'scale_re_init': 0.9,
               're_init_index': None,
               're_init_threshold': 0.2,
               'buffer_size_plast': 200,
               'noise_weight': 30.0,
               'variance_th_c': 0.5,
               'variance_th_p': 0.4,
               'learning_rate': 0.007,
               'inh_learning_rate': 0.01,
               'decay': 150,
               'tau_stdp': 10 * ms,
               'tau_pred': 1.5 * ms,
               'seed' : 42,
               }


# Dictionaries for mismatch

mismatch_neuron_param = {
    'Inoise': 0,
    'Iconst': 0,
    'kn': 0,
    'kp': 0,
    'Ut': 0,
    'Io': 0,
    'Cmem': 0.2,
    'Iath': 0.2,
    'Iagain': 0.2,
    'Ianorm': 0.2,
    'Ica': 0.2,
    'Itauahp': 0.2,
    'Ithahp': 0.2,
    'Cahp': 0.2,
    'Ishunt': 0.2,
    'Ispkthr': 0.2,
    'Ireset': 0.2,
    'Ith': 0.2,
    'Itau': 0.2,
    'refP': 0.2,
}

mismatch_synap_param = {
    'I_syn': 0,
    'kn_syn': 0,
    'kp_syn': 0,
    'Ut_syn': 0,
    'Csyn': 0.2,
    'I_tau': 0.2,
    'I_th': 0.2,
    'Io_syn': 0.2,
    'w_plast': 0,
    'baseweight': 0.2,
}
