"""
Created on Mon May 03 2021

@author=pabloabur

This file contains parameters of the building block related to quantized
stochastic models described by Wang et al. (2018).

"""
from brian2 import ms, mV

connection_probability_HS19 = {'pyr_pyr': 0.10, # TODO dendrite
                               'pyr_pv': 0.45,
                               'pyr_sst': 0.35,
                               'pyr_vip': 0.10,
                               'pv_pyr': 0.60,
                               'pv_pv': 0.50,
                               'sst_pv': 0.60,
                               'sst_pyr': 0.55, # TODO dendrite
                               'sst_vip': 0.45,
                               'vip_sst': 0.50,
                               # I made up values below
                               'input_pyr': 1.0,
                               'input_pv': 1.0,
                               'input_sst': 1.0,
                               'input_vip': 1.0
                               }

connection_probability_old = {'pyr_pyr': 1., # TODO dendrite
                              'pyr_pv': 0.5,
                              'pyr_sst': 0.,
                              'pyr_vip': 0.,
                              'pv_pyr': 0.60,
                              'pv_pv': 0.10,
                              'sst_pv': 0.,
                              'sst_pyr': 0., # TODO dendrite
                              'sst_vip': 0.,
                              'vip_sst': 0.,
                              'input_pyr': 1.0,
                              'input_pv': 1.0,
                              'input_sst': 1.0,
                              'input_vip': 1.0
                              }

connection_probability = {'pyr_pyr': 1., # TODO dendrite
                          'pyr_pv': 0.15,
                          'pyr_sst': 0.15,
                          'pyr_vip': 0.10,
                          'pv_pyr': 1.0,
                          'pv_pv': 1.0,
                          'sst_pv': 0.9,
                          'sst_pyr': 1.0, # TODO dendrite
                          'sst_vip': 0.9,
                          'vip_sst': 0.65,
                          'input_pyr': 1.0,
                          'input_pv': 1.0,
                          'input_sst': 1.0,
                          'input_vip': 1.0
                          }

excitatory_neurons = {'tau': 20*ms,
                      'Vm': 3*mV,
                      'thr_max': 15*mV
                     }

inhibitory_neurons = {'tau': 10*ms,
                      'Vm': 3*mV
                     }

excitatory_plastic_synapse = {'tausyn': 5*ms,
                              'taupre': 20*ms,
                              'taupost': 20*ms,
                              'rand_num_bits_Apre': 4,
                              'rand_num_bits_Apost': 4,
                              'stdp_thres': 1
                              }

inhibitory_synapse = {'tausyn': 10*ms
                     }

synapse_mean_weight = {'e_i': 3, # 1
                       'i_e': 4, # 1
                       'e_e': 1,
                       'i_i': 1,
                       'inp_e': 3,
                       'inp_i': 1 # 2
                       }

mismatch_neuron_param = {'tau': 0.1  # 0.2
                         }

mismatch_synapse_param = {'tausyn': 0.1  # 0.2
                        }
mismatch_plastic_param = {'taupre': 0.1,  # 0.2
                          'taupost': 0.1  # 0.2
                          }
