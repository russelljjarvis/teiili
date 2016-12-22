# -*- coding: utf-8 -*-
'''
This file specifies equations for neurons and synapses used in neural networks

author: Daniele Conti
Mail: daniele.conti@polito.it
date: 8th dec 2016

'''

#IMPORTS 

import numpy as np
from brian2 import *
import ParametersNet as p

#NEURONS EQUATIONS-------------------------------------------

eqs=Equations('''
                 dImem/dt = (Ipos - Imem * (1 + Iahp / Itau)) / ( taum * (1 + Ith / (Imem + noise + Io)) ) : amp
                 Ipos =  ( (Ith / Itau) * (Iin - Iahp - Itau)  + Ifb ) : amp
                 Ifb  =  ( (Ia / Itau) * (Imem + Ith) ) : amp
                 Ia   =  ( Iagain * 1 / (1 + exp(-(Imem - Iath)/ Ianorm) ) ) : amp
       
                 dIahp/dt=(Iposa - Iahp ) / tauahp : amp
                 dIca/dt = (Iposa - Ica ) / tauca : amp

                 Iin = Iin_ex + Iin_inh + Iin_teach : amp
                 Iin_ex : amp
                 Iin_inh : amp
                 Iin_teach : amp

                 mu = 0.25 * pA : amp
                 sigma = 0.1 * pA : amp
                 b = sign(2 * rand() -1) : 1 (constant over dt)
                 noise = b * (sigma * randn() + mu) : amp (constant over dt)
                 '''
                 )
 
#-----------------------------------------------------
#SYNAPSES EQUATIONS
#----------------------------------------------------

Inhibitory_model='''
                 w:1
                 dIsyn/dt = (-Isyn) / tauinhib : amp (event-driven) 
                 Iin_inh_post = Isyn : amp (summed)
                 '''

Inhibitory_output_model='''
                 w:1
                 dIsyn/dt = (- Isyn) / tauinhib : amp (event-driven)
                 Iin_inh_post = Isyn : amp (summed)
                 '''

Teacher_model='''
                 w : 1
                 dIsyn/dt = (Iwexc - Isyn) / tauexc : amp (event-driven) 
                 Iin_teach_post = Isyn : amp (summed)
                 '''
#-----------------------------------------------------
#SYNAPSES PRE-EVENT Instructions
#----------------------------------------------------


Memristive_nonplastic_pre='''
                 Isyn += Iw * w
                 '''

Inhibitory_pre='''
                 Isyn += Iw_in_h * w
                 '''

Inhibitory_output_pre='''
                 Isyn += Iwinh * w
                 '''

Teacher_pre='''
                 Isyn += Iwexc * w
                 '''
#-----------------------------------------------------
#SYNAPSES POST-EVENT Instructions
#----------------------------------------------------


