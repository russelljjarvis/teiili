'''
This file contains default parameters for different synapse equations as a dict.
use tools.setparams(group,paramdict) to set them to a synapse

Created on 17.12.2016

@author: Alpha (renner.alpha@gmail.com)
@author: Daniele Conti (daniele.conti@polito.it)

'''

from brian2 import *

fusiSyn_default = {"w_plus": 0.2,
                   "w_minus": 0.2,
                   "theta_upl": 180 * mV,
                   "theta_uph": 1 * volt,
                   "theta_downh": 90 * mV,
                   "theta_downl": 50 * mV,
                   "theta_V": -59 * mV,
                   "alpha": 0.0001 / second,
                   "beta": 0.0001 / second,
                   "tau_ca": 8 * ms,
                   "w_ca": 250 * mV,
                   "w_min": 0,
                   "w_max": 1,
                   "theta_w": 0.5,
                   "w": 0,
                   "gWe": 7 * nS,
                   "gIe": 0 * nS,
                   "taugIe": 5 * ms,
                   "EIe": 60.0 * mV}  # maybe not good default params, just placeholder!

simpleSyn_default = {
                    "tau": 5 * ms,
                    "Iw": 1 * nA}

revSyn_default = {"gIe": 0 * nS,
                  "gIi": 0 * nS,
                  "taugIe": 5 * ms,
                  "taugIi": 6 * ms,
                  "EIe": 60.0 * mV,
                  "EIi": -90.0 * mV,
                  "gWe": 7 * nS,
                  "gWi": -3 * nS}

alphaSynapse_default = {"gWe": 0.7 * nS, #EPSG peak amplitude
                  "tau": 2 * ms,
                  "Vrev" : 0 * mV,
                  "t_spike" : 1*second
                  }


StdpSyn_default = {"gIe": 0 * nS,
                   "taugIe": 5 * ms,
                   "EIe": 60.0 * mV,
                   "gWe": 7 * nS,
                   "taupre": 20 * ms,
                   "taupost": 20 * ms,
                   "w_max": 0.01,
                   "diffApre": 0.01,
                   "Q_diffAPrePost": 1.05,
                   "weight": 1,
                   "w": 0.005}


fusiMemristor = {
    "Iwca": 0.25 * pA,
    "theta_pl": 4 * pA,
    "theta_dl": 0 * pA,
    "theta_pu": 9.99 * pA,
    "theta_du": 3.99 * pA,
    "Imemthr": 0.05 * nA,
    "Iw_fm": 0.5 * nA,
    "tau_fm": 5 * msecond,
    "prop": 2.0,
    "R_min": 3000,
    "R_max": 8000,
    "R0_d": 3234,
    "R0_p": 6031.56,
    "alpha_p": 3350,
    "alpha_d": 502,
    "beta_p": 0.612,
    "C_p": 286,
    "D_p": 17,
    "D_d": 139,
    "count_up": 0,
    "count_down": 0}  # good default params only for single neuron example!

DefaultExcitatorySynapseP = {
    "tauexc": 5 * msecond,
    "Iw_exc": 5. * pamp}

DefaultInhibitorySynapseP = {
    "tauinhib": 5 * msecond,
    "Iw_inh": 0.5 * namp}

DefaultTeacherSynapseP = {
    "taut": 5 * msecond,
    "Iw_t": 1.0 * namp}


Braderfusi = {
    "Iwca": 0.25 * pA,
    "theta_pl": 4 * pA,
    "theta_dl": 0 * pA,
    "theta_pu": 9.99 * pA,
    "theta_du": 3.99 * pA,
    "Imemthr": 0.05 * nA,
    "Iw_fm": 0.5 * nA,
    "tau_fm": 5 * msecond,
    "alpha": 3.5 * Hz,
    "beta": 3.5 * Hz,
    "wa": 0.1,
    "wb": 0.1,
    "wth": 0.5,
    "count_up": 0,
    "count_down": 0}  # good default params only for single neuron example!
KernelP = {
    "tau": 40 * msecond,
    "omega": 30 / second,
    "sigma_gaussian": 20 * msecond}

SiliconSynP = {
    "Vth": 0.8 * volt,  # should be close to Vdd
    "Vtau": 7 * mV,  # calculated by assuming a tau ~ 10 * ms
    "Vdd": 1.8 * volt,
    "Csyn": 0.1 * pF,
    "Io": 0.5 * pA,
    "Ut": 25 * mV,
    "kn": 0.75,
    "kp": 0.66,
    "duration": 10 * msecond}
