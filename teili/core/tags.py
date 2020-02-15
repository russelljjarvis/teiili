#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 19:13:06 2019

@author: matteo

Collection of dictionaries to initialize tags in a group specific manner.

Naming convention as follows: basic_[buildingblock]_[group_name]

"""

basic_wta_n_exc = {'mismatch': False,
                   'noise': False,
                   'level': 1,
                   'sign': 'exc',
                   'target sign': 'None',
                   'num_inputs': 3,
                   'bb_type': 'WTA',
                   'group_type': 'Neuron',
                   'connection_type': 'None',
                   }


basic_wta_n_inh = {'mismatch': False,
                   'noise': False,
                   'level': 1,
                   'sign': 'inh',
                   'target sign': 'None',
                   'num_inputs': 2,
                   'bb_type': 'WTA',
                   'group_type': 'Neuron',
                   'connection_type': 'None',
                   }

basic_wta_n_sg = {'mismatch': False,
                  'noise': False,
                  'level': 1,
                  'sign': 'None',
                  'target sign': 'None',
                  'num_inputs': 0,
                  'bb_type': 'WTA',
                  'group_type': 'SpikeGenerator',
                  'connection_type': 'None',
                  }

basic_wta_s_exc_exc = {'mismatch': False,
                       'noise': False,
                       'level': 1,
                       'sign': 'exc',
                       'target sign': 'exc',
                       'num_inputs': 0,
                       'bb_type': 'WTA',
                       'group_type': 'Connection',
                       'connection_type': 'rec',
                       }

basic_wta_s_exc_inh = {'mismatch': False,
                       'noise': False,
                       'level': 1,
                       'sign': 'exc',
                       'target sign': 'inh',
                       'num_inputs': 0,
                       'bb_type': 'WTA',
                       'group_type': 'Connection',
                       'connection_type': 'rec',
                       }

basic_wta_s_inh_exc = {'mismatch': False,
                       'noise': False,
                       'level': 1,
                       'sign': 'inh',
                       'target sign': 'exc',
                       'num_inputs': 0,
                       'bb_type': 'WTA',
                       'group_type': 'Connection',
                       'connection_type': 'rec',
                       }

basic_wta_s_inh_inh = {'mismatch': False,
                       'noise': False,
                       'level': 1,
                       'sign': 'inh',
                       'target sign': 'inh',
                       'num_inputs': 0,
                       'bb_type': 'WTA',
                       'group_type': 'Connection',
                       'connection_type': 'rec',
                       }

basic_wta_s_inp_exc = {'mismatch': False,
                       'noise': False,
                       'level': 1,
                       'sign': 'None',
                       'target sign': 'exc',
                       'num_inputs': 0,
                       'bb_type': 'WTA',
                       'group_type': 'Connection',
                       'connection_type': 'ff',
                       }

basic_octa_s_proj_pred = {'mismatch': True,
                          'noise': False,
                          'level': 2,
                          'sign': 'exc',
                          'target sign': 'exc',
                          'num_inputs': 0,
                          'bb_type': 'OCTA',
                          'group_type': 'Connection',
                          'connection_type': 'ff',
                          }

basic_octa_s_pred_proj = {'mismatch': False,
                          'noise': False,
                          'level': 2,
                          'sign': 'exc',
                          'target sign': 'exc',
                          'num_inputs': 0,
                          'bb_type': 'OCTA',
                          'group_type': 'Connection',
                          'connection_type': 'fb',
                          }

basic_octa_n_sg = {'mismatch': False,
                   'noise': False,
                   'level': 2,
                   'sign': 'exc',
                   'target sign': 'None',
                   'num_inputs': 0,
                   'bb_type': 'OCTA',
                   'group_type': 'SpikeGenerator',
                   'connection_type': 'None',
                   }

basic_octa_n_proj = {'mismatch': True,
                     'noise': False,
                     'level': 2,
                     'sign': 'exc',
                     'target sign': 'exc',
                     'num_inputs': 2,
                     'bb_type': 'OCTA',
                     'group_type': 'Neurons',
                     'connection_type': 'None',
                     }

basic_octa_s_pred_noise = {'mismatch': False,
                           'noise': True,
                           'level': 2,
                           'sign': 'exc',
                           'target sign': 'exc',
                           'num_inputs': 0,
                           'bb_type': 'OCTA',
                           'group_type': 'Connection',
                           'connection_type': 'rec',
                           }

basic_octa_s_comp_noise = {'mismatch': False,
                           'noise': True,
                           'level': 2,
                           'sign': 'exc',
                           'target sign': 'exc',
                           'num_inputs': 0,
                           'bb_type': 'OCTA',
                           'group_type': 'Connection',
                           'connection_type': 'rec',
                           }

basic_octa_pred_noise_sg = {'mismatch': False,
                            'noise': True,
                            'level': 2,
                            'sign': 'exc',
                            'target sign': 'None',
                            'num_inputs': 0,
                            'bb_type': 'OCTA',
                            'group_type': 'SpikeGenerator',
                            'connection_type': 'None',
                            }


basic_octa_comp_noise_sg = {'mismatch': False,
                            'noise': True,
                            'level': 2,
                            'sign': 'exc',
                            'target sign': 'None',
                            'num_inputs': 0,
                            'bb_type': 'OCTA',
                            'group_type': 'SpikeGenerator',
                            'connection_type': 'None',
                            }

basic_threeway_1WTA_to_2WTA = {'mismatch' : 0,
                           'noise' : 0,
			               'level': 2,
			               'sign': 'exc',
			               'target sign': 'exc',
			               'num_inputs' : 0,
			               'bb_type' : 'Threeway',
			               'group_type' : 'Connection',
			               'connection_type' : 'ff',
				            }

basic_threeway_2WTA_to_1WTA = {'mismatch' : 0,
                           'noise' : 0,
			               'level': 2,
			               'sign': 'exc',
			               'target sign': 'exc',
			               'num_inputs' : 0,
			               'bb_type' : 'Threeway',
			               'group_type' : 'Connection',
			               'connection_type' : 'fb',
				            }


basic_tags_empty = {'mismatch' : 0,
                    'noise' : 0,
		            'level': 0 ,
		            'sign': 'None',
		            'target sign': 'None',
		            'num_inputs' : 0,
		            'bb_type' : 'None',
		            'group_type' : 'None',
		            'connection_type' : 'None',
	    	        }
