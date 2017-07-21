'''
This is just an example
can be deleted later on, when more sophisticated examples are available

can be used as inspiration for unit test of new neuron equations 

@author: Alpha
'''

#import matplotlib
import matplotlib.pyplot as plt
from brian2 import ms,mV,pA,Network,prefs,SpikeGeneratorGroup,NeuronGroup, \
                   Synapses,SpikeMonitor,StateMonitor,figure, plot,show,xlabel,ylabel
from brian2 import *
import numpy as np
import time
#import sys
#from os import path
#sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from NCSBrian2Lib.neuronEquations import ExpAdaptIF as neuronEq
from NCSBrian2Lib.synapseEquations import simpleSyn as synapseEq # reversalSynV as synapseEq
from NCSBrian2Lib.Parameters.neuronParams import gerstnerExpAIFdefaultregular as neuronPar
from NCSBrian2Lib.Parameters.synapseParams import simpleSyn_default as synapsePar # revSyndefault as synapsePar
from NCSBrian2Lib.tools import setParams, fkernel1d, printStates

prefs.codegen.target = "numpy" 


eqsDict = neuronEq(debug=True)

testNet = Network()

tsSeq = np.asarray(range(0,600,2)) * ms
indSeq = np.concatenate((np.zeros(50, dtype=np.int), np.ones(50, dtype=np.int),
                        np.zeros(50, dtype=np.int), np.ones(50, dtype=np.int),
                        np.zeros(50, dtype=np.int), 2*np.ones(50, dtype=np.int) ))
gSeqInpGroup = SpikeGeneratorGroup(3, indices = indSeq, times=tsSeq)

#print(eqsDict['model'])
gSeqGroup = NeuronGroup(3, **eqsDict)
#gSeqGroup = NeuronGroup(3, refractory=5*ms, method = "euler", **eqsDict)
#sDict = fusiSyn(debug = True)
#sDict = reversalSynV(debug = True)
sDict = synapseEq(debug = True)
synInpSeqe = Synapses(gSeqInpGroup, gSeqGroup, method = "euler", **sDict)
synInpSeqe.connect('i==j') 
synInpSeqe.weight = 1

#===============================================================================
# synSeqSeq1e = Synapses(gSeqGroup,   gSeqGroup,    method = "euler", **sDict)
# synSeqSeq1e.connect('True') 
# synSeqSeq1e.weight = -1
# setParams(synSeqSeq1e ,synapsePar)
# testNet.add((synSeqSeq1e))
#===============================================================================
             
setParams(synInpSeqe ,synapsePar)
#setParams(synInpSeqe ,revSyndefault)
#setParams(synInpSeqe ,fusiDefault, debug=True)
setParams(gSeqGroup ,neuronPar)
gSeqGroup.refP = 5*ms
#synInpSeqe.w = 1

spikemonSeq = SpikeMonitor(gSeqGroup)
spikemonSeqInp = SpikeMonitor(gSeqInpGroup)
statemonSeq = StateMonitor(gSeqGroup,('Vm','Ie','Ii'), record=[0,1,2])
#statemonSyn = StateMonitor(synInpSeqe,('Iesyn'), record=[0,1,2])

testNet.add((gSeqInpGroup,gSeqGroup,spikemonSeq,spikemonSeqInp,statemonSeq,synInpSeqe))#,statemonSyn))

start = time.clock()
duration = 600 * ms
testNet.run(duration)
end = time.clock()
print ('simulation took ' + str(end - start) + ' sec')
print('done!')


fig = figure(figsize=(8,3))
plot(statemonSeq.t/ms, statemonSeq[0].Vm/mV, label='V')
plot(statemonSeq.t/ms, statemonSeq[1].Vm/mV, label='V')
plot(statemonSeq.t/ms, statemonSeq[2].Vm/mV, label='V')
xlabel('Time [ms]')
ylabel('V (mV)')
#plt.show()#savefig('fig/figSeqV.png')

#===============================================================================
# fig = figure(figsize=(8,3))
# plot(statemonSyn.t/ms, statemonSyn.gIe[0]/nS, label='gIe')
# plot(statemonSyn.t/ms, statemonSyn.gIe[1]/nS, label='gIe')
# plot(statemonSyn.t/ms, statemonSyn.gIe[2]/nS, label='gIe')
# xlabel('Time [ms]')
# ylabel('gIe (nS)')
#===============================================================================

#===============================================================================
# fig = figure(figsize=(8,3))
# plot(statemonSyn.t/ms, statemonSyn.Iesyn[0]/nS, label='Iesyn')
# plot(statemonSyn.t/ms, statemonSyn.Iesyn[1]/nS, label='Iesyn')
# plot(statemonSyn.t/ms, statemonSyn.Iesyn[2]/nS, label='Iesyn')
# xlabel('Time [ms]')
# ylabel('Iesyn')
#===============================================================================


fig = figure(figsize=(8,3))
plot(statemonSeq.t/ms, statemonSeq.Ie[0]/pA, label='Ie')
plot(statemonSeq.t/ms, statemonSeq.Ie[1]/pA, label='Ie')
plot(statemonSeq.t/ms, statemonSeq.Ie[2]/pA, label='Ie')
xlabel('Time [ms]')
ylabel('Ie (pA)')


fig = figure(figsize=(8,3))
plot(statemonSeq.t/ms, statemonSeq.Ii[0]/pA, label='Ii')
plot(statemonSeq.t/ms, statemonSeq.Ii[1]/pA, label='Ii')
plot(statemonSeq.t/ms, statemonSeq.Ii[2]/pA, label='Ii')
xlabel('Time [ms]')
ylabel('Ii (pA)')
show()#savefig('fig/figSeqIe.png')


