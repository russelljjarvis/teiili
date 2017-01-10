# -*- coding: utf-8 -*-
'''

author: Daniele Conti
Mail: daniele.conti@polito.it
date: 8th dec 2016

'''



#IMPORTS 

import numpy as np
from brian2 import *

from Training import Training
from synapseEquations import *
from neuronEquations import Silicon
#from NCSBrian2Lib.tools import *
from tools import *
from synapseParams import *
from neuronParams import *

#-----------------------------------------------------
#CREATING NEURONS GROUPS
#-----------------------------------------------------

eqSil = Silicon()


InputNeurons = PoissonGroup(1, 100 * Hz)
OutputNeurons = NeuronGroup(1, method = 'euler', refractory = 0.5 * ms, **eqSil)
InhibitoryNeurons = NeuronGroup(1, method = 'euler', refractory = 0.5 * ms, **eqSil)
TeacherNeurons = PoissonGroup(1, rates = 0 * Hz)

setParams(OutputNeurons, SiliconNeuronP, debug = True)
setParams(InhibitoryNeurons, SiliconNeuronP, debug = False)
#-----------------------------------------------------
#CREATING SYNAPSES GROUPS
#-----------------------------------------------------
sdict = MemristiveFusiSynapses(debug = True)
sdict_inh = DefaultInhibitorySynapses()
sdict_inh_out = DefaultInhibitorySynapses(inh2output=True)
sdict_t = DefaultTeacherSynapses()

Input_Output_train = Synapses(InputNeurons, OutputNeurons, method = 'euler', **sdict)

Inhibitory_Input = Synapses(InputNeurons, InhibitoryNeurons, method = 'euler', **sdict_inh)

Inhibitory_Output = Synapses(InhibitoryNeurons, OutputNeurons, method = 'euler', **sdict_inh_out)
 
Teacher_Output = Synapses(TeacherNeurons, OutputNeurons, method = 'euler', **sdict_t)

#-----------------------------------------------------
#CONNECTION AND INIT
#-----------------------------------------------------

Input_Output_train.connect(True) #fully connected
Inhibitory_Input.connect(True) #fully connected
Inhibitory_Output.connect(True) #fully connected
Teacher_Output.connect(True) #fully connected

Input_Output_train.w = 1.2
Inhibitory_Input.w = 0.35
Inhibitory_Output.w = 1
Teacher_Output.w = 1

setParams(Input_Output_train, fusiMemristor, debug = True)
setParams(Inhibitory_Input, DefaultInhibitorySynapseP, debug = True)
setParams(Inhibitory_Output, DefaultInhibitorySynapseP, debug = False)
setParams(Teacher_Output, DefaultTeacherSynapseP, debug = True)

#RUN Simulation----------------------

training_img_time = 200 * ms
resting_time = 10 * ms
print "start new training"
    
weights, timestamps, SM, SpikeMI, SpikeMO, StateMIsyn, SMInh, StateMInh, SMIsyninh = Training(InputNeurons, OutputNeurons, Input_Output_train, InhibitoryNeurons, Inhibitory_Input, Inhibitory_Output, TeacherNeurons, Teacher_Output, training_img_time, resting_time)

figure()
subplot(611)
title('SPK input (blue), SPK out (red), SPK inh (green)')
plot(SpikeMI.t/ms, SpikeMI.i, 'b.')
plot(SpikeMO.t/ms, SpikeMO.i, 'r.')
plot(SMInh.t/ms, SMInh.i, 'g.')
#xlim([0, 1400])
subplot(612)
#plot(timestamps[:], 2 * np.ones(len(timestamps[:])))
title('weights')
plot(timestamps[:], weights[0][:])
subplot(613)
title('Iin_ex excit')
plot(StateMIsyn.t[:], StateMIsyn.Isyn[0])
subplot(614)
title('Imem output neuron (blu) + Iin total (red)')
plot(timestamps[:], SM.Imem[0,:], 'b')
plot(timestamps[:], SM.Iin[0,:], 'r')
plot(timestamps[:], np.zeros(len(timestamps[:])), 'k')
subplot(615)
title('Ica output neuron')
plot(timestamps[:], SM.Ica[0,:])
subplot(616)
title('Imem Inhibitory neuron (blu) + Iin_inh (red)')
plot(timestamps[:], StateMInh.Imem[0,:], 'b')
plot(timestamps[:], SMIsyninh.Isyn[0,:], 'r')

figure()
subplot(511)
title('Output Neuron variables: Imem')
plot(timestamps[:], SM.Imem[0,:], 'b')
subplot(512)
title('Ipos')
plot(timestamps[:], SM.Ipos[0,:], 'b')
subplot(513)
title('Ifb')
plot(timestamps[:], SM.Ifb[0,:], 'b')
subplot(514)
title('Ia')
plot(timestamps[:], SM.Ia[0,:], 'b')
subplot(515)
title('Iahp')
plot(timestamps[:], SM.Iahp[0,:], 'b')

figure()
subplot(211)
title('Noise on (Imem + Io)')
plot(timestamps[:], SM.noise[0,:], '.b')
subplot(212)
title('Noise histogram')
hist(SM.noise[0,:], 100)


show()






