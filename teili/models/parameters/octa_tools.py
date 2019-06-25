#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:19:57 2019

@author: matteo
"""

def add_octa_mismatch():
 #Todo:  Transofo    
   # compressionWTA._groups['spike_gen'].add_mismatch(octa_param.mismatch_neuron_param,seed=42)
    
    compressionWTA._groups['spike_gen'].add_mismatch(octa_param.mismatch_neuron_param,seed=42)
    compressionWTA._groups['n_exc'].add_mismatch(octa_param.mismatch_neuron_param,seed=43)
    compressionWTA._groups['n_inh'].add_mismatch(octa_param.mismatch_neuron_param,seed=44)
    
    compressionWTA._groups['s_inp_exc'].add_mismatch(octa_param.mismatch_synap_param,seed=45)
    compressionWTA._groups['s_exc_exc'].add_mismatch(octa_param.mismatch_synap_param,seed=46)
    compressionWTA._groups['s_exc_inh'].add_mismatch(octa_param.mismatch_synap_param,seed=47)
    compressionWTA._groups['s_inh_exc'].add_mismatch(octa_param.mismatch_synap_param,seed=48)
    
    predictionWTA._groups['n_exc'].add_mismatch(octa_param.mismatch_neuron_param,seed=49)
    predictionWTA._groups['n_inh'].add_mismatch(octa_param.mismatch_neuron_param,seed=50)
    
    predictionWTA._groups['s_inp_exc'].add_mismatch(octa_param.mismatch_synap_param,seed=51)
    predictionWTA._groups['s_exc_exc'].add_mismatch(octa_param.mismatch_synap_param,seed=52)
    predictionWTA._groups['s_exc_inh'].add_mismatch(octa_param.mismatch_synap_param,seed=53)
    predictionWTA._groups['s_inh_exc'].add_mismatch(octa_param.mismatch_synap_param,seed=1337)

    error_connection.add_mismatch(octa_param.mismatch_synap_param,seed=54)

    return None

    # RUN REGULARLY FUNCTIONS ###############
# WEIGHT DECAY
def add_weight_decay(params):

    if params  == 'global':
        add_weight_decay(compressionWTA._groups['s_inp_exc'],
                         decay='global',
                         learning_rate=params['learning_rate'])
    
        add_weight_decay(compressionWTA._groups['s_exc_exc'],
                         decay='global',
                         learning_rate=params['learning_rate'])
        
        add_weight_decay(predictionWTA._groups['s_inp_exc'],
                         decay='global',
                         learning_rate=params['learning_rate'])
    
        add_weight_decay(predictionWTA._groups['s_exc_exc'],
                         decay='global',
                         learning_rate=params['learning_rate'])
        
    
        add_weight_decay(error_connection,
                         decay='global',
                         learning_rate=params['learning_rate'])
        
        add_pred_weight_decay(prediction_connection,
                         decay='global',
                         learning_rate=params['learning_rate'])
        
        
        return None

def add_weight_re_init():

    add_re_init_weights(compressionWTA._groups['s_inp_exc'], 
                        re_init_threshold=octa_param.octaParams['re_init_threshold'],
                        dist_param_re_init=octa_param.octaParams['dist_param_re_init'], 
                        scale_re_init=octa_param.octaParams['scale_re_init'],
                        distribution=octa_param.octaParams['distribution'])


    add_re_init_weights(compressionWTA._groups['s_exc_exc'], 
                        re_init_threshold=octa_param.octaParams['re_init_threshold'],
                        dist_param_re_init=octa_param.octaParams['dist_param_re_init'], 
                        scale_re_init=octa_param.octaParams['scale_re_init'],
                        distribution=octa_param.octaParams['distribution'])


    add_re_init_weights(predictionWTA._groups['s_inp_exc'], 
                        re_init_threshold=octa_param.octaParams['re_init_threshold'],
                        dist_param_re_init=octa_param.octaParams['dist_param_re_init'], 
                        scale_re_init=octa_param.octaParams['scale_re_init'],
                        distribution=octa_param.octaParams['distribution'])



    add_re_init_weights(predictionWTA._groups['s_exc_exc'], 
                        re_init_threshold=octa_param.octaParams['re_init_threshold'],
                        dist_param_re_init=octa_param.octaParams['dist_param_re_init'], 
                        scale_re_init=octa_param.octaParams['scale_re_init'],
                        distribution=octa_param.octaParams['distribution'])

    add_re_init_weights(error_connection, 
                        re_init_threshold=octa_param.octaParams['re_init_threshold'],
                        dist_param_re_init=octa_param.octaParams['dist_param_re_init'], 
                        scale_re_init=octa_param.octaParams['scale_re_init'],
                        distribution=octa_param.octaParams['distribution'])


    add_re_init_ipred(prediction_connection, 
                      re_init_threshold=octa_param.octaParams['re_init_threshold'])
    
    return None

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



