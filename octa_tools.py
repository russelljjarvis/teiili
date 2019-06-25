#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:19:57 2019

@author: matteo
"""
from teili.models.parameters import octa_param 


from teili.octa.interfaces.lock_and_load import *

from teili.octa.tools.add_run_reg import *
from teili.octa.tools.add_mismatch import *
from teili.octa.tools.weight_init import weight_init

def add_bb_mismatch(bb):
    '''
    This allows to add mismatch to all the neuron and connection groups present in a building block
    
    Args:
        bb (Building Block Object, required)

    Returns:
        None
    '''
    for i in bb._groups:
        if bb._groups[i]._tags['group_type'] == 'Neuron':
            bb._groups[i].add_mismatch(octa_param.mismatch_neuron_param, seed=42)
            bb._groups[i]._tags['mismatch'] = 1
        elif bb._groups[i]._tags['group_type'] == 'Connection':
            bb._groups[i].add_mismatch(octa_param.mismatch_synap_param, seed= 42)
            bb._groups[i]._tags['mismatch'] = 1
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


def monitor():
    return 0

def save_monitor(monitor,  filename, path):
    '''Save SpikeMonitor using numpy.save()'''
    date = time.strftime("%d_%m_%Y")
    if not os.path.exists(path + date):
        os.mkdir(path + date + "/")
    path = path + date + "/"
    filename = time.strftime("%d_%m_%H_%M") + '_' + filename
    toSave = np.zeros((2, len(monitor.t))) * np.nan
    toSave[0, :] = np.asarray(monitor.i)
    toSave[1, :] = np.asarray(monitor.t /ms)
    np.save(path + filename, toSave)
    
def load_monitor(filename):
    '''Load a saved spikemonitor using numpy.load()'''
    data  = np.load(filename)
    monitor()
    monitor.i = data[0, :]
    monitor.t = data[1, :] * ms
    return monitor

def save_weights(weights, filename, path):
    '''Save weight matrix between to populations into .npy file'''
    date = time.strftime("%d_%m_%Y")
    if not os.path.exists(path + date):
        os.mkdir(path + date + "/")
    path = path + date + "/"
    filename = time.strftime("%d_%m_%H_%M") + '_' + filename
    toSave = np.zeros(np.shape(weights)) * np.nan
    toSave = weights
    np.save(path + filename, toSave)
    
def save_params(params, filename, path):
    '''Save dictionary containing neuron/synapse paramters or simulation parameters'''
    date = time.strftime("%d_%m_%Y")
    if not os.path.exists(path + date):
        os.mkdir(path + date + "/")
    path = path + date + "/"
    filename = time.strftime("%d_%m_%H_%M") + '_' + filename
    np.save(path + filename, params)
    
def load_weights(filename=None, nrows=None, ncols=None):
    '''Load weights from .npy file'''
    if filename is not None:
        weight_matrix = np.load(filename)
    else:
        filename = filedialog.askopenfilename()
        weight_matrix = np.load(filename)
    
    if nrows is None and ncols is None:
        raise UserWarning('Please specify the dimension of the matrix, since it not squared.')
    if ncols is None:
        weight_matrix = weight_matrix.reshape((nrows, -1))
    elif nrows is None:
        weight_matrix = weight_matrix.reshape((-1, ncols))
    else:
        weight_matrix = weight_matrix.reshape((nrows, ncols))
    return weight_matrix



