from brian2 import *
import numpy as np


def setParams(neurongroup , params):
    for par in params:
        setattr(neurongroup, par , params[par])


# function that calculates 1D index from 2D index
@implementation('numpy', discard_units=True)
@check_units(x0=1,x1=1,n2dNeurons=1,result=1)
def xy2ind(x0,x1,n2dNeurons):
    return int(x0)+int(x1)*n2dNeurons

# function that calculates 2D index from 1D index
@implementation('numpy', discard_units=True)
@check_units(ind=1,n2dNeurons=1,result=1)
def ind2xy(ind,n2dNeurons):
    ret = (np.mod(np.round(ind),n2dNeurons), np.floor_divide(np.round(ind),n2dNeurons))     
    return ret