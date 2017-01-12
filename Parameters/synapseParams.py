'''
This file contains default parameters for different synapse equations as a dict.
use tools.setparams(group,paramdict) to set them to a synapse

Created on 17.12.2016

@author: Alpha (renner.alpha@gmail.com)
@author: Daniele Conti (daniele.conti@polito.it)

'''

from brian2 import *

fusidefault = { "w_plus" : 0.2,
                "w_minus": 0.2, 
                "theta_upl" : 180 *mV,
                "theta_uph" : 1*volt,  
                "theta_downh" : 90 *mV,
                "theta_downl" : 50 *mV, 
                "theta_V" : -59 *mV, 
                "alpha" : 0.0001/second, 
                "beta" : 0.0001/second, 
                "tau_ca" : 8*ms,
                "w_ca" : 250*mV, 
                "qe_plast": 340 *pA,
                "w" : 0} #maybe not good default params, just placeholder!


fusiMemristor = { 
                "Iwca" : 0.25 *pA,
                "theta_pl" : 4 *pA,
                "theta_dl" : 0 *pA,
                "theta_pu" : 9.99 *pA,
                "theta_du" : 3.99 *pA,
                "Imemthr" : 0.05 *nA,
                "Iw_fm" : 0.5 *nA, 
                "tau_fm" : 5 *msecond,
                "prop" : 2.0,
                "R_min" : 3000,
                "R0_d" : 3234,
                "R0_p" : 6031.56,
                "alpha_p" : 3350,
                "alpha_d" : 502,
                "beta_p" : 0.612,
                "C_p" : 286,
                "D_p" : 17,
                "D_d" : 139} #good default params only for single neuron example!

DefaultInhibitorySynapseP = {
                "tauinhib" : 5 *msecond,
                "Iw_inh"  : 0.5 *namp}

DefaultTeacherSynapseP = {
                "tauexc" : 5 *msecond,
                "Iwexc"  : 0.5 *namp}
