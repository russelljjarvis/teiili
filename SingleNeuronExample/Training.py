# -*- coding: utf-8 -*-
"""
Created on Tue Nov 1 2016

@author: Daniele Conti
"""
import numpy as np
from brian2 import *
import random
import ParametersNet as p

def Training(InputNeurons, OutputNeurons, Input_Output_train, InhibitoryNeurons, Inhibitory_Input, Inhibitory_Output, TeacherNeurons, Teacher_Output):

    #PARAMETERS-----------------------------------
    #Neurons
    Ispkthr = p.Ispkthr
    Ispkthr_inh = p.Ispkthr_inh
    Ireset = p.Ireset

    Itau = p.Itau
    taum = p.taum
    Ith = p.Ith
    Io = p.Io
    Iagain = p.Iagain
    Iath = p.Iath
    Ianorm = p.Ianorm
    Iposa = p.Iposa
    tauca = p.tauca
    tauahp = p.tauahp
    
    #Synapses
    tausyn = p.tausyn
    tauinhib = p.tauinhib
    Iwinh = p.Iwinh
    tauexc = p.tauexc
    Iwexc = p.Iwexc
    tauexc = p.tauexc
    Iw_in_h = p.Iw_in_h
    Iw = p.Iw


    #MONITORS--------------------------------------
    SM = StateMonitor(OutputNeurons, True, record = True)
    SpikeMI = SpikeMonitor(InputNeurons)
    SpikeMO = SpikeMonitor(OutputNeurons)
    SpikeMInh = SpikeMonitor(InhibitoryNeurons)
    StateMInh = StateMonitor(InhibitoryNeurons, True, record = True)
    StateMw = StateMonitor(Input_Output_train, 'w', record = True)
    StateMIsyn = StateMonitor(Input_Output_train, 'Isyn', record = True)
    StateMIsyninh = StateMonitor(Inhibitory_Output, 'Isyn', record = True)

    #TRAINING
    count = 0

    while count < 5:

        run(p.training_img_time)

        Input_Output_train.Isyn=0
        rates = None

        run(p.resting_time)
        count += 1 
    
    weights = StateMw.w
    timestamps = StateMw.t

    print weights
    print timestamps

    return weights, timestamps, SM, SpikeMI, SpikeMO, StateMIsyn, SpikeMInh, StateMInh, StateMIsyninh


