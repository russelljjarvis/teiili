'''
This file contains default parameters for different synapse equations as a dict.
use tools.setparams(group,paramdict) to set them to a synapse

Created on 17.12.2016

@author: Alpha (renner.alpha@gmail.com)
@author: Daniele Conti (daniele.conti@polito.it)

'''

from brian2 import *

fusidefault = { "w_plus": 0.2,
                "w_minus": 0.2, 
                "theta_upl": 180 *mV,
                "theta_uph": 1*volt,  
                "theta_downh": 90 *mV,
                "theta_downl": 50 *mV, 
                "theta_V": -59 *mV, 
                "alpha": 0.0001/second, 
                "beta": 0.0001/second, 
                "tau_ca": 8*ms,
                "w_ca": 250*mV, 
                "qe_plast": 340 *pA,
                "w": 0} #maybe not good default params, just placeholder!
