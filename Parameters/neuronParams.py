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
