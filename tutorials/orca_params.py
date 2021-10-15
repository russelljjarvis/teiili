"""
Created on Mon May 03 2021

@author=pabloabur

This file contains parameters of the building block related to quantized
stochastic models described by Wang et al. (2018).

"""
from brian2 import ms, mV, mA

# Synapses/connections
connection_probability_HS19 = {'pyr_pyr': 0.50,
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

connection_probability = {'pyr_pyr': 0.5,
                          'pyr_pv': 0.15,
                          'pyr_sst': 0.15,
                          'pyr_vip': 0.10,
                          'pv_pyr': 1.0,
                          'pv_pv': 1.0,
                          'sst_pv': 0.9,
                          'sst_pyr': 1.0,
                          'sst_vip': 0.9,
                          'vip_sst': 0.65,
                          # I made up values below
                          'ff_pyr': 1.0,
                          'ff_pv': 1.0,
                          'ff_sst': 1.0,
                          'ff_vip': 0.0,
                          'fb_pyr': 1.0,
                          'fb_pv': 0.0,
                          'fb_sst': 0.0,
                          'fb_vip': 1.0
                          }

intralaminar_conn_prob = {'L23': {'pyr_pyr': 0.5,
                                  'pyr_pv': 0.15, #TODO
                                  'pyr_sst': 0.15, #TODO
                                  'pyr_vip': 0.10, #TODO
                                  'pv_pyr': 1.0, #TODO
                                  'pv_pv': 1.0, #TODO
                                  'sst_pv': 0.9, #TODO
                                  'sst_pyr': 1.0, #TODO
                                  'sst_vip': 0.9, #TODO
                                  'vip_sst': 0.65}, #TODO
                          'L4': {'pyr_pyr': 0.5,
                                 'pyr_pv': 0.15, #TODO
                                 'pyr_sst': 0.15, #TODO
                                 'pyr_vip': 0.10, #TODO
                                 'pv_pyr': 1.0, #TODO
                                 'pv_pv': 1.0, #TODO
                                 'sst_pv': 0.9, #TODO
                                 'sst_pyr': 1.0, #TODO
                                 'sst_vip': 0.9, #TODO
                                 'vip_sst': 0.65}, #TODO
                          'L5': {'pyr_pyr': 0.2,
                                 'pyr_pv': 0.15, #TODO
                                 'pyr_sst': 0.15, #TODO
                                 'pyr_vip': 0.10, #TODO
                                 'pv_pyr': 1.0, #TODO
                                 'pv_pv': 1.0, #TODO
                                 'sst_pv': 0.9, #TODO
                                 'sst_pyr': 1.0, #TODO
                                 'sst_vip': 0.9, #TODO
                                 'vip_sst': 0.65}, #TODO
                          'L6': {'pyr_pyr': 0.2,
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

interlaminar_conn_prob = {'L23_L4': {'pyr_pyr': 0.03,
                                     'pyr_pv': 0.15, #TODO
                                     'pyr_sst': 0.15, #TODO
                                     'pyr_vip': 0.10, #TODO
                                     'pv_pyr': 1.0, #TODO
                                     'pv_pv': 1.0, #TODO
                                     'sst_pv': 0.9, #TODO
                                     'sst_pyr': 1.0, #TODO
                                     'sst_vip': 0.9, #TODO
                                     'vip_sst': 0.65}, #TODO
                          'L23_L5': {'pyr_pyr': 0.04,
                                     'pyr_pv': 0.15, #TODO
                                     'pyr_sst': 0.15, #TODO
                                     'pyr_vip': 0.10, #TODO
                                     'pv_pyr': 1.0, #TODO
                                     'pv_pv': 1.0, #TODO
                                     'sst_pv': 0.9, #TODO
                                     'sst_pyr': 1.0, #TODO
                                     'sst_vip': 0.9, #TODO
                                     'vip_sst': 0.65}, #TODO
                          'L23_L6': {'pyr_pyr': 0.03,
                                     'pyr_pv': 0.15, #TODO
                                     'pyr_sst': 0.15, #TODO
                                     'pyr_vip': 0.10, #TODO
                                     'pv_pyr': 1.0, #TODO
                                     'pv_pv': 1.0, #TODO
                                     'sst_pv': 0.9, #TODO
                                     'sst_pyr': 1.0, #TODO
                                     'sst_vip': 0.9, #TODO
                                     'vip_sst': 0.65}, #TODO
                          'L4_L23': {'pyr_pyr': 0.05,
                                     'pyr_pv': 0.15, #TODO
                                     'pyr_sst': 0.15, #TODO
                                     'pyr_vip': 0.10, #TODO
                                     'pv_pyr': 1.0, #TODO
                                     'pv_pv': 1.0, #TODO
                                     'sst_pv': 0.9, #TODO
                                     'sst_pyr': 1.0, #TODO
                                     'sst_vip': 0.9, #TODO
                                     'vip_sst': 0.65}, #TODO
                          'L4_L5': {'pyr_pyr': 0.01,
                                    'pyr_pv': 0.15, #TODO
                                    'pyr_sst': 0.15, #TODO
                                    'pyr_vip': 0.10, #TODO
                                    'pv_pyr': 1.0, #TODO
                                    'pv_pv': 1.0, #TODO
                                    'sst_pv': 0.9, #TODO
                                    'sst_pyr': 1.0, #TODO
                                    'sst_vip': 0.9, #TODO
                                    'vip_sst': 0.65}, #TODO
                          'L4_L6': {'pyr_pyr': 0.02,
                                    'pyr_pv': 0.15, #TODO
                                    'pyr_sst': 0.15, #TODO
                                    'pyr_vip': 0.10, #TODO
                                    'pv_pyr': 1.0, #TODO
                                    'pv_pv': 1.0, #TODO
                                    'sst_pv': 0.9, #TODO
                                    'sst_pyr': 1.0, #TODO
                                    'sst_vip': 0.9, #TODO
                                    'vip_sst': 0.65}, #TODO
                          'L5_L23': {'pyr_pyr': 0.03,
                                     'pyr_pv': 0.15, #TODO
                                     'pyr_sst': 0.15, #TODO
                                     'pyr_vip': 0.10, #TODO
                                     'pv_pyr': 1.0, #TODO
                                     'pv_pv': 1.0, #TODO
                                     'sst_pv': 0.9, #TODO
                                     'sst_pyr': 1.0, #TODO
                                     'sst_vip': 0.9, #TODO
                                     'vip_sst': 0.65}, #TODO
                          'L5_L4': {'pyr_pyr': 0.0,
                                    'pyr_pv': 0.15, #TODO
                                    'pyr_sst': 0.15, #TODO
                                    'pyr_vip': 0.10, #TODO
                                    'pv_pyr': 1.0, #TODO
                                    'pv_pv': 1.0, #TODO
                                    'sst_pv': 0.9, #TODO
                                    'sst_pyr': 1.0, #TODO
                                    'sst_vip': 0.9, #TODO
                                    'vip_sst': 0.65}, #TODO
                          'L5_L6': {'pyr_pyr': 0.01,
                                    'pyr_pv': 0.15, #TODO
                                    'pyr_sst': 0.15, #TODO
                                    'pyr_vip': 0.10, #TODO
                                    'pv_pyr': 1.0, #TODO
                                    'pv_pv': 1.0, #TODO
                                    'sst_pv': 0.9, #TODO
                                    'sst_pyr': 1.0, #TODO
                                    'sst_vip': 0.9, #TODO
                                    'vip_sst': 0.65}, #TODO
                          'L6_L4': {'pyr_pyr': 0.1,
                                    'pyr_pv': 0.15, #TODO
                                    'pyr_sst': 0.15, #TODO
                                    'pyr_vip': 0.10, #TODO
                                    'pv_pyr': 1.0, #TODO
                                    'pv_pv': 1.0, #TODO
                                    'sst_pv': 0.9, #TODO
                                    'sst_pyr': 1.0, #TODO
                                    'sst_vip': 0.9, #TODO
                                    'vip_sst': 0.65}, #TODO
                          'L6_L23': {'pyr_pyr': 0.0,
                                     'pyr_pv': 0.15, #TODO
                                     'pyr_sst': 0.15, #TODO
                                     'pyr_vip': 0.10, #TODO
                                     'pv_pyr': 1.0, #TODO
                                     'pv_pv': 1.0, #TODO
                                     'sst_pv': 0.9, #TODO
                                     'sst_pyr': 1.0, #TODO
                                     'sst_vip': 0.9, #TODO
                                     'vip_sst': 0.65}, #TODO
                          'L6_L5': {'pyr_pyr': 0.0,
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

excitatory_synapse_soma = {'tausyn': 5*ms,
                           'gain_syn': 1*mA,
                           'delay': 0*ms,
                           'taupre': 20*ms,
                           'taupost': 30*ms,
                           'rand_num_bits_Apre': 4,
                           'rand_num_bits_Apost': 4,
                           'stdp_thres': 1
                           }

excitatory_synapse_dend = {'tausyn': 5*ms,
                           'gain_syn': 1*mA,
                           'delay': 0*ms,
                           'taupre': 20*ms,
                           'taupost': 30*ms,
                           'rand_num_bits_Apre': 4,
                           'rand_num_bits_Apost': 4,
                           'stdp_thres': 1
                           }

inhibitory_synapse_soma = {'tausyn': 10*ms,
                           'gain_syn': 1*mA,
                           'delay': 0*ms
                          }

inhibitory_synapse_dend = {'tausyn': 10*ms,
                           'gain_syn': 1*mA,
                           'delay': 0*ms
                          }

synapse_mean_weight = {'e_i': 3,
                       'i_e': 1,
                       'e_e': 4,
                       'i_i': 1,
                       'inp_e': 2,
                       'inp_i': 2
                       }

# Neurons/populations
excitatory_neurons = {'tau': 20*ms,
                      'Vm': 3*mV,
                      'thr_max': 15*mV,
                      'Vm_noise': 0*mV
                     }

inhibitory_neurons = {'tau': 20*ms,
                      'Vm': 3*mV,
                      'thr_max': 15*mV,
                      'Vm_noise': 0*mV
                     }

inhibitory_ratio = {'L23': {'pv': .37,
                            'sst': .20,
                            'vip': .43},
                    # TODO use previous values
                    'L4': {'pv': .68,#1,#
                           'sst': .20,#.02,#
                           'vip': .12},#.02},#
                    'L5': {'pv': .52,
                           'sst': .37,
                           'vip': .11},
                    'L6': {'pv': .49,
                           'sst': .38,
                           'vip': .13}
                    }

exc_pop_proportion = {'L23': 1.2,
                      'L4': 1,
                      'L5': 1.1,
                      'L6': 2
                      }

ei_ratio = {'L23': 4,
            'L4': 4,
            'L5': 6,
            'L6': 6
            }

# Heterogeneity
mismatch_neuron_param = {'tau': 0.1
                         }

mismatch_synapse_param = {'tausyn': 0.1
                        }
mismatch_plastic_param = {'taupre': 0.1,
                          'taupost': 0.1
                          }
