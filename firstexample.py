'''
This is just an example
can be deleted later on, when more sophisticated examples are available

can be used as inspiration for unit test of new neuron equations 

@author: Alpha
'''

import matplotlib
import matplotlib.pyplot as plt
from brian2 import *
import numpy as np
import time
from NCSBrian2Lib.neuronEquations import ExpAdaptIF, ExpAdaptIFReversal
from NCSBrian2Lib.tools import *
from NCSBrian2Lib.Parameters.neuronParams import *

prefs.codegen.target = "numpy" 

VT1 = -49*mV
eqsDict = ExpAdaptIF(tauw = 120*ms,DeltaT=3*mV,VT='VT1',debug=True)
#eqsDict = ExpAdaptIFReversal(debug=True)

testNet = Network()

tsSeq = np.asarray(range(0,600,2)) * ms
indSeq = np.concatenate((np.zeros(50, dtype=np.int), np.ones(50, dtype=np.int),
                        np.zeros(50, dtype=np.int), np.ones(50, dtype=np.int),
                        np.zeros(50, dtype=np.int), 2*np.ones(50, dtype=np.int) ))
gSeqInpGroup = SpikeGeneratorGroup(3, indices = indSeq, times=tsSeq)

gSeqGroup = NeuronGroup(3, **eqsDict, refractory=1*ms, method = "euler")
synInpSeqe = Synapses(gSeqInpGroup, gSeqGroup, on_pre = 'Ie += 665*pA')
#synInpSeqe = Synapses(gSeqInpGroup, gSeqGroup, on_pre = 'gIe += 5*nS')
synInpSeqe.connect('i==j') 

setParams(gSeqGroup , gerstnerExpAIFdefaultregular)
#setParams(gSeqGroup ,gerstnerExpAIFReversaldefaultregular)

spikemonSeq = SpikeMonitor(gSeqGroup)
spikemonSeqInp = SpikeMonitor(gSeqInpGroup)
statemonSeq = StateMonitor(gSeqGroup,('Vm','Ie'), record=[0,1,2])
#statemonSeq = StateMonitor(gSeqGroup,('Vm','gIe'), record=[0,1,2])

testNet.add((gSeqInpGroup,gSeqGroup,spikemonSeq,spikemonSeqInp,statemonSeq,synInpSeqe))

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
plt.show()#savefig('fig/figSeqV.png')

fig = figure(figsize=(8,3))
plot(statemonSeq.t/ms, statemonSeq.Ie[0]/pA, label='Ie')
plot(statemonSeq.t/ms, statemonSeq.Ie[1]/pA, label='Ie')
plot(statemonSeq.t/ms, statemonSeq.Ie[2]/pA, label='Ie')
#plot(statemonSeq.t/ms, statemonSeq.gIe[0]/pA, label='Ie')
#plot(statemonSeq.t/ms, statemonSeq.gIe[1]/pA, label='Ie')
#plot(statemonSeq.t/ms, statemonSeq.gIe[2]/pA, label='Ie')
xlabel('Time [ms]')
ylabel('Ie (pA)')
plt.show()#savefig('fig/figSeqIe.png')


