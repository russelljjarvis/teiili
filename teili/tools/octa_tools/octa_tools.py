#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:19:57 2019

@author: matteo
"""
from teili.tools.octa_tools import octa_param 

from teili.tools.run_regularly.add_run_reg import add_weight_decay,\
add_pred_weight_decay, add_re_init_weights, add_weight_regularization,\
add_re_init_ipred, add_activity_proxy
from teili.tools.octa_tools.weight_init import weight_init


def add_bb_mismatch(bb):
    '''
    This allows to add mismatch to all the neuron and connection groups present in a building block
    
    Args:
        bb (Building Block Object, required)

    Returns:
        None
    '''
                 
    for i in bb.groups:
        if bb.groups[i]._tags['group_type'] == 'Neuron':
            bb.groups[i].add_mismatch(octa_param.mismatch_neuron_param, seed=42)
            bb.groups[i]._tags['mismatch'] = 1
        elif bb.groups[i]._tags['group_type'] == 'Connection':
            bb.groups[i].add_mismatch(octa_param.mismatch_synap_param, seed= 42)
            bb.groups[i]._tags['mismatch'] = 1
        else:
            pass
    return None


def add_decay_weight(group, decay_strategy, learning_rate):
        '''
    This allows to add a weight decay run regular function following a pre defined
    decay strategay
    
    Args:
        groups (list, required) : list of groups that implemet

    Returns:
        None
        ''' 
        
        for grp in group:
          
            add_weight_decay(grp, decay_strategy, learning_rate)
        
        
        return None


def add_weight_pred_decay(group, decay_strategy, learning_rate):

        for grp in group: 
            add_pred_weight_decay(grp, decay_strategy, learning_rate)
        return None

def add_weight_re_init(group, re_init_threshold, dist_param_re_init, scale_re_init,
                       distribution):

    for grp in group:
        add_re_init_weights(grp, 
                        re_init_threshold=re_init_threshold,
                        dist_param_re_init=dist_param_re_init, 
                        scale_re_init=scale_re_init,
                        distribution=distribution)

    return None


def add_weight_re_init_ipred(group, re_init_threshold):
    
    for grp in group:
        add_re_init_ipred(grp,  re_init_threshold=re_init_threshold)
    
    return None

def add_regulatization_weight(group, buffer_size):
    for grp in group:
        add_weight_regularization(grp,buffer_size=buffer_size)
        
        
def add_proxy_activity(group, buffer_size, decay):
    for grp in group:
        add_activity_proxy(grp,
                   buffer_size=buffer_size,
                   decay=decay)

        
def add_weight_init (group , dist_param, scale, distribution):    
    for grp in group:
        grp.w_plast = weight_init(grp, 
                                   dist_param=dist_param, 
                                   scale=scale,
                                   distribution=distribution)
        
        


        
        