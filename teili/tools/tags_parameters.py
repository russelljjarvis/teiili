#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 19:13:06 2019

@author: matteo
"""

'''
Tags for the WTA

'''

basic_tags_n_exc =  {'mismatch' : 0,
              'level': 1 ,     
              'sign': 'exc',
              'target sign': 'None',
              'num_inputs' : 3,
              'bb_type' : 'WTA',
              'group_type' : 'Neuron',
              'connection_type' : 'None',
              }
 
  
basic_tags_n_inh =   { 'mismatch' : 0,
             'level': 1 ,     
              'sign': 'inh',
              'target sign': 'None',
              'num_inputs' : 2,
              'bb_type' : 'WTA',
              'group_type' : 'Neuron',
              'connection_type' : 'None',
    }
 
basic_tags_n_sg =   { 'mismatch' : 0,
             'level': 1 ,     
              'sign': 'None',
              'target sign': 'None',
              'num_inputs' : 0,
              'bb_type' : 'WTA',
              'group_type' : 'SPikeGenerator',
              'connection_type' : 'None',
    }

 
basic_tags_s_exc_exc =   { 'mismatch' : 0,
             'level': 1 ,     
              'sign': 'exc',
              'target sign': 'exc',
              'num_inputs' : 0,
              'bb_type' : 'WTA',
              'group_type' : 'Connection',
              'connection_type' : 'rec',
    }    


basic_tags_s_exc_inh =   { 'mismatch' : 0,
             'level': 1 ,     
              'sign': 'exc',
              'target sign': 'inh',
              'num_inputs' : 0,
              'bb_type' : 'WTA',
              'group_type' : 'Connection',
              'connection_type' : 'rec',
    }    



basic_tags_s_inh_exc =   { 'mismatch' : 0,
             'level': 1 ,     
              'sign': 'inh',
              'target sign': 'exc',
              'num_inputs' : 0,
              'bb_type' : 'WTA',
              'group_type' : 'Connection',
              'connection_type' : 'rec',
    }    


basic_tags_s_inh_inh =   { 'mismatch' : 0,
             'level': 1 ,     
              'sign': 'inh',
              'target sign': 'inh',
              'num_inputs' : 0,
              'bb_type' : 'WTA',
              'group_type' : 'Connection',
              'connection_type' : 'rec',
    }    


basic_tags_s_inp_exc =   { 'mismatch' : 0,
             'level': 1 ,     
              'sign': 'None',
              'target sign': 'exc',
              'num_inputs' : 0,
              'bb_type' : 'WTA',
              'group_type' : 'Connection',
              'connection_type' : 'rec',
    }    

basic_tags_compression_con_octa = {
         'mismatch' : 0,
             'level': 2 ,     
              'sign': 'exc',
              'target sign': 'exc',
              'num_inputs' : 0,
              'bb_type' : 'OCTA',
              'group_type' : 'Connection',
              'connection_type' : 'fb',
    }    


    
basic_tags_prediction_con_octa = {
         'mismatch' : 0,
             'level': 2 ,     
              'sign': 'exc',
              'target sign': 'exc',
              'num_inputs' : 0,
              'bb_type' : 'OCTA',
              'group_type' : 'Connection',
              'connection_type' : 'fb',
    }    


basic_tags_empty =   { 'mismatch' : 0,
             'level': 0 ,     
              'sign': 'None',
              'target sign': 'None',
              'num_inputs' : 0,
              'bb_type' : 'None',
              'group_type' : 'None',
              'connection_type' : 'None',
    }    




