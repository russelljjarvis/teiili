'''
Created 03.2017
@author: Alpha
'''
import numpy as np
import time
from brian2 import *
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from NCSBrian2Lib.BasicBuildingBlocks.WTA import gen2dWTA,gen1dWTA
from NCSBrian2Lib.Plotting.WTAplot import plotWTA,plotWTATiles
from tools import xy2ind

#===============================================================================
prefs.codegen.target = "numpy" 

#defaultclock.dt = 50 *us

exampleNet = Network()

nWTA1dNeurons = 10
nWTA2dNeurons = 20
nPerGroup = 1

duration = 600 * ms

#===============================================================================
## create wta

(gwtaExaGroup,gwtaExaInhGroup,gwtaExaInpGroup,
synInpwtaExa1e,synwtaExawtaExa1e,synwtaExaInh1e,synInhwtaExa1i,
spikemonwtaExa,spikemonwtaExaInh,spikemonwtaExaInp,statemonwtaExa) = gen2dWTA('wtaExa',nNeurons = nWTA2dNeurons,debug = True,
             weInpWTA = 1.8, weWTAInh =3.0, wiInhWTA = -1, w_lat=1.7,
             rpWTA = 2.5*ms, rpInh = 1*ms,
             sigm = 2.2, nInhNeurons = 5, cutoff = 9)
            
exampleNet.add((gwtaExaGroup,gwtaExaInhGroup,gwtaExaInpGroup,synInpwtaExa1e,synwtaExawtaExa1e,synwtaExaInh1e,synInhwtaExa1i,
            spikemonwtaExa,spikemonwtaExaInh,spikemonwtaExaInp,statemonwtaExa))



#xindwtaExa = np.concatenate((2*np.ones(60),3*np.ones(60)))
#tswtaExa = np.concatenate((range(20,80),range(30,90)))
#gwtaExaInpGroup.set_spikes(np.asarray(xindwtaExa), np.asarray(tswtaExa)*ms)

xindwtaExa = np.concatenate((  np.asarray([xy2ind(5,5,nWTA2dNeurons) for i in range(30)]),
                               np.asarray([xy2ind(12,12,nWTA2dNeurons) for i in range(30)]),
                               np.asarray([xy2ind(ii,ii,nWTA2dNeurons) for ii in range(nWTA2dNeurons) for i in range(5)])
                               #np.asarray([xy2ind(ix,round(abs(ix-10)**1.5),nWTA2dNeurons) for ix in range(nWTA2dNeurons) for i in range(5)])
                           ))
tswtaExa = np.concatenate((range(0,60,2),range(100,160,2),range(200,600,4)))
gwtaExaInpGroup.set_spikes(np.asarray(xindwtaExa), np.asarray(tswtaExa)*ms)


#from NCSBrian2Lib.synapseEquations import reversalSynV as synapseEq
#from NCSBrian2Lib.Parameters.neuronParams import gerstnerExpAIFdefaultregular as neuronPar
#from NCSBrian2Lib.Parameters.synapseParams import revSyndefault as synapsePar
#from NCSBrian2Lib.tools import setParams
#synapseEqDict = synapseEq(debug = True)

#gwtaExaInpGroup2 = PoissonGroup(nWTA1dNeurons, 200 * Hz,name = 'Inp2')
#synInp2 = Synapses(gwtaExaInpGroup2, gwtaExaGroup,    method = "euler", **synapseEqDict)
#synInp2.connect('i==j')
#setParams(synInp2, synapsePar, debug = False)
#synInp2.weight = 1
#exampleNet.add((gwtaExaInpGroup2,synInp2))


#===============================================================================
## simulation
start = time.clock()
exampleNet.run(duration)
end = time.clock()
print ('simulation took ' + str(end - start) + ' sec')
print('simulation done!')
print(spikemonwtaExa.i)
#print(synInpwtaExa1e.weight)
#print(synwtaExaInh1e.weight)
#print(synInhwtaExa1i.weight)
#print(synwtaExawtaExa1e.weight)

plotWTA('wtaExa',duration,nWTA2dNeurons,True,spikemonwtaExa,spikemonwtaExaInh,spikemonwtaExaInp,statemonwtaExa)

## WTA plot tiles over time
plotWTATiles('wtaExa',duration,nWTA2dNeurons, spikemonwtaExa,interval=10*ms,savepath=False, showfig = False)
show()
print('done!')