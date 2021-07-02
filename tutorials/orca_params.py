"""
Created on Mon May 03 2021

@author=pabloabur

This file contains parameters of the building block related to quantized
stochastic models described by Wang et al. (2018).

"""
from brian2 import ms, mV, mA

connection_probability_HS19 = {'pyr_pyr': 0.10,
                               'pyr_pv': 0.45,
                               'pyr_sst': 0.35,
                               'pyr_vip': 0.10,
                               'pv_pyr': 0.60,
                               'pv_pv': 0.50,
                               'sst_pv': 0.60,
                               'sst_pyr': 0.55,
                               'sst_vip': 0.45,
                               'vip_sst': 0.50,
                               # I made up values below
                               'ff_pyr': 1.0,
                               'ff_pv': 1.0,
                               'ff_sst': 1.0,
                               'ff_vip': 0.0,
                               'fb_pyr': 0.1,
                               'fb_pv': 0.0,
                               'fb_sst': 0.0,
                               'fb_vip': 0.8
                               }

connection_probability_old = {'pyr_pyr': 1.,
                              'pyr_pv': 0.5,
                              'pyr_sst': 0.,
                              'pyr_vip': 0.,
                              'pv_pyr': 0.60,
                              'pv_pv': 0.10,
                              'sst_pv': 0.,
                              'sst_pyr': 0.,
                              'sst_vip': 0.,
                              'vip_sst': 0.,
                              'ff_pyr': 1.0,
                              'ff_pv': 1.0,
                              'ff_sst': 0.0,
                              'ff_vip': 0.0,
                              'fb_pyr': 0.8,
                              'fb_pv': 0.0,
                              'fb_sst': 0.0,
                              'fb_vip': 0.8
                              }

connection_probability = {'pyr_pyr': 1.,
                          'pyr_pv': 0.15,
                          'pyr_sst': 0.15,
                          'pyr_vip': 0.10,
                          'pv_pyr': 1.0,
                          'pv_pv': 1.0,
                          'sst_pv': 0.9,
                          'sst_pyr': 1.0,
                          'sst_vip': 0.9,
                          'vip_sst': 0.65,
                          'ff_pyr': 1.0,
                          'ff_pv': 1.0,
                          'ff_sst': 1.0,
                          'ff_vip': 0.0 # or 1.0 if pPE 
                          }

excitatory_neurons = {'tau': 20*ms,
                      'Vm': 3*mV,
                      'thr_max': 15*mV,
                      'Vm_noise': 0*mV
                     }

inhibitory_neurons = {'tau': 10*ms,
                      'Vm': 3*mV,
                      'Vm_noise': 0*mV
                     }

excitatory_synapse_soma = {'tausyn': 5*ms,
                           'gain_syn': 1*mA,
                           'delay': 0*ms,
                           'taupre': 20*ms,
                           'taupost': 20*ms,
                           'rand_num_bits_Apre': 4,
                           'rand_num_bits_Apost': 4,
                           'stdp_thres': 1
                           }

excitatory_synapse_dend = {'tausyn': 3*ms,
                           'gain_syn': 0.5*mA,
                           'delay': 1*ms,
                           'taupre': 20*ms,
                           'taupost': 20*ms,
                           'rand_num_bits_Apre': 4,
                           'rand_num_bits_Apost': 4,
                           'stdp_thres': 1
                           }

inhibitory_synapse_soma = {'tausyn': 10*ms,
                           'gain_syn': 1*mA,
                           'delay': 0*ms
                          }

inhibitory_synapse_dend = {'tausyn': 7*ms,
                           'gain_syn': 0.5*mA,
                           'delay': 1*ms
                          }

synapse_mean_weight = {'e_i': 3, # 1
                       'i_e': 1, # 1
                       'e_e': 4,
                       'i_i': 1,
                       'inp_e': 4,
                       'inp_i': 1 # 2
                       }

mismatch_neuron_param = {'tau': 0.1  # 0.2
                         }

mismatch_synapse_param = {'tausyn': 0.1  # 0.2
                        }
mismatch_plastic_param = {'taupre': 0.1,  # 0.2
                          'taupost': 0.1  # 0.2
                          }
