'''
This module will provide different types of standard weight matrix for 2 neuron population
Type of weight matrices which are supported:

Inputs:
    save_path:          Path where to store the weightMatrix. If not set explicitly matrix stored
                        in your temporary folder depending on you OS
    population1:        Brian2 neurongroup
    population2:        Brian2 neurongroup
    mode:               Specifies from which type of distribution weights are sampled
                        Mode can be set to 'normal', 'exponential', 'int', 'uniform'
                        Default: 'normal'
    mu:                 Mean of distribution needed for 'normal' distributed random weights.
                        Default: 0
    sigma:              Standard deviation of distribution needed for 'normal' distributed random weights.
                        Default: 0.25.
    weightRange:        Maximum and minimum value th weights can get. Used by 'int' and 'uniform' mode
    connectionType:     Specifies which type of connectivity is between two populations desired. \
                        Can be either 'fully' (fully connected) or 'sparse'
    connectivityMatrix: If connectionType is sparse, the actual connectivityMatrix needs to be passed to this function
    save:               Flag whether to save created weightMatrix in predefined save_path (see above).
                        If not set weightMatrix will be returned


Output: If save is set:
                Weight matrix will be stored in save_path
        if not:
                Weight matrix will be returned

Random weight matrix
Uniformly distributed
differentiate between fully connected and sparsely connected populations

Author: Moritz Milde
Date: 02.12.2016
'''

import numpy as np
from brian2 import *
import os
import tempfile


class generateWeightMatrix():
    def __init__(self, save_path=None):
        '''
        Init function of the class
        If save_path is not specified but save flag is set (see below) weight matrix
        will be stored in your temporary directory
        '''
        if save_path is not None:
            self.save_path = save_path
            # check if save_path is already created
            self.save_path = os.path.join(save_path, "brian2", "weightMatrices")
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
        else:
            self.save_path = tempfile.gettempdir()
        self.weightRange = 70
        self.mu = 0
        self.sigma = 0.25

    def getWeights(self):
        '''
        This function generates random weights between two neuron populations specified outside
        Based on the selected mode the random distribution is differently initialized
        Supported modes:
            - 'normal' normal distributed
            - 'int' randomly distributed integers from -range to range
            - 'exponential' weights are sampled from an exponential distribution
            - 'uniform' uniformly spaced weights between -range and range
        '''
        if self.mode == 'normal':
            weightMatrix = np.random.normal(self.mu, self.sigma, size=self.matrixSize)
        elif self.mode == 'int':
            weightMatrix = np.random.randint(-self.weightRange, self.weightRange, self.matrixSize)
        elif self.mode == 'exponential':
            weightMatrix = np.random.exponential(size=self.matrixSize)
        elif self.mode == 'uniform':
            weightMatrix = np.random.uniform(-self.weightRange, self.weightRange, self.matrixSize)
        else:
            raise Exception("%s mode is not supported! Please add the respective function if needed." % (self.mode))
        return weightMatrix

    def randomWeightMatrix(self, population1, population2, mode='normal', mu=None, sigma=None, weightRange=None,
                           connectionType='fully', connectivityMatrix=None, save=False):
        if mu is not None:
            self.mu = mu
        if sigma is not None:
            self.sigma = sigma
        if weightRange is not None:
            self.weightRange = weightRange
        self.matrixSize = (population1.N, population2.N)
        self.mode = mode
        if connectionType == 'fully':
            weightMatrix = getWeights()
            if save:
                np.save(self.save_path + '/', weightMatrix)
            else:
                return weightMatrix
        # only if we don't have fc populations we need connectivity matrix
        elif connectionType == 'sparse':
            assert(type(connectivityMatrix) is np.ndarray), 'You want a sparse connectivity pattern,\nplease pass the connection matrix'
            self.matrixSize = np.shape(connectivityMatrix)
            weightMatrix = getWeights()
            if save:
                np.save(self.save_path + '/', weightMatrix)
            else:
                return weightMatrix
