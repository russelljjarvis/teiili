'''
This file contains default parameters for different neuron equations as a dict.
use tools.setparams(group,paramdict) to set them to a neurongroup

Created on 17.12.2016

@author: Alpha (renner.alpha@gmail.com)
'''

from brian2 import *


gerstnerExpAIFdefaultregular = {"tauIe"  : 5     *ms,
                                "tauIi"  : 7     *ms,
                                "C"      : 281   *pF,
                                "gL"     : 35    *nS,
                                "EL"     : -70.6 *mV,
                                "VT"     : -50.4 *mV,
                                "DeltaT" : 2     *mV,
                                "tauw"   : 144   *ms,
                                "a"      : 4     *nS,
                                "b"      : 0.0805*nA,
                                "Vr"     : -70.6 *mV,
                                "Vm"     : -70.6 *mV}
               
gerstnerExpAIFdefaultbursting ={"tauIe"  : 5     *ms,
                                "tauIi"  : 7     *ms,
                                "C"      : 281   *pF,
                                "gL"     : 35    *nS,
                                "EL"     : -70.6 *mV,
                                "VT"     : -50.4 *mV,
                                "DeltaT" : 2     *mV,
                                "tauw"   : 20    *ms,
                                "a"      : 4     *nS,
                                "b"      : 0.5   *nA,
                                "Vr"     : -45.4 *mV,
                                "Vm"     : -70.6 *mV}

gerstnerExpAIFdefaultfast   =  {"tauIe"  : 5     *ms,
                                "tauIi"  : 7     *ms,
                                "C"      : 281   *pF,
                                "gL"     : 35    *nS,
                                "EL"     : -70.6 *mV,
                                "VT"     : -50.4 *mV,
                                "DeltaT" : 2     *mV,
                                "tauw"   : 144   *ms,
                                "a"      : 3.9   *nS,
                                "b"      : 0     *nA,
                                "Vr"     : -70.6 *mV,
                                "Vm"     : -70.6 *mV} 

gerstnerExpAIFReversaldefaultregular = {"taugIe"  : 5     *ms,
                                "taugIi"  : 7     *ms,
                                "C"      : 281   *pF,
                                "gL"     : 35    *nS,
                                "EL"     : -70.6 *mV,
                                "VT"     : -50.4 *mV,
                                "DeltaT" : 2     *mV,
                                "tauw"   : 144   *ms,
                                "a"      : 4     *nS,
                                "b"      : 0.0805*nA,
                                "Vr"     : -70.6 *mV,
                                "Vm"     : -70.6 *mV,
                                "EIe"    : 60.0 *mV,
                                "EIi"    : -90.0  *mV}

SiliconNeuronP = {
#--------------------------------------------------------
# VLSI process parameters
#--------------------------------------------------------
                                "kn"     : 0.75,
                                "kp"     : 0.66,
                                "Ut"     : 25 * mV,
                                "Io"     : 0.5 * pA,
# Silicon neuron parameters                      
                                "Csyn"   : 0.1 * pF,
                                "Cmem"   : 0.5 * pF,
                                "Cahp"   : 0.5 * pF,
# Fitting parameters
                                "Iagain" : 1 * nA,
                                "Iath"   : 20 * nA,
                                "Ianorm" : 1 * nA,
#---------------------------------------------------------
#Adaptative and Calcium parameters
#---------------------------------------------------------
                                "tauca"  : 40 * msecond, #Calcium spike decay rate
                                "Iposa"  : 0.3 * pA,
                                "Iwa"    : 0 * pA, #Adaptation spike amplitude
                                "Itaua"  : 1 * pA,
#---------------------------------------------------------
# Neuron parameters
#---------------------------------------------------------
                                "Ispkthr" : 20 * nA, #Spike threshold of excitatory neurons
                                "Ispkthr_inh" : 5 * nA, #Spike threshold of inhibitory
                                "Ireset" : 1 * pA, #Reset Imem to Ireset after each spike 
                                "Ith"    : 1 * pA,
                                "Itau"   : 1 * pA,
                                "Ithreshold" : 0.15 * nA}
