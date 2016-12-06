'''
This module will provide different types of standard weight matrix for 2 neuron population
Type of weight matrices which are supported:

Random weight matrix
Uniformly distributed
differentiate between fully connected and sparsely connected populations

Author: Moritz Milde
Date: 02.12.2016
'''

import numpy as np
from brian2 import *
import os


class generateWeightMatrix():
    def __init__(self, weightRange=70, save_path='/media/moritz/Data/'):
        self.save_path = save_path
        # check if save_path is already created
        self.save_path = os.path.join(save_path, "brian2", "weightMatrices")
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        self.range = weightRaneg

    def randomWeightMatrix(self, population1, population2, connectionType='fully', connectivityMatrix=None, save=False):
        matrixSize = (population1.N, population2.N)
        if connectionType == 'fully':
            weightMatrix = np.random.randint(-self.weightRange, self.weightRange, matrixSize)
            if save:
                np.save(self.save_path + '', weightMatrix)
            else:
                return weightMatrix
        # only if we don't have fc populations we need connectivity matrix
        elif connectionType == 'sparse':
            assert(type(connectivityMatrix) is np.ndarray), 'If you have a sparse connectivity pattern,\nplease pass the connection matrix'
            matrixSize = np.shape(connectivityMatrix)
            weightMatrix = np.random.randint(-self.weightRange, self.weightRange, matrixSize)
            if save:
                np.save(self.save_path + '', weightMatrix)
            else:
                return weightMatrix
