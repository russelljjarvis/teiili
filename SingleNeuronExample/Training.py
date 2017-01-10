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
    StateMR = StateMonitor(Input_Output_train, 'R_new_p', record = True)
    StateMRS = StateMonitor(Input_Output_train, 'noiseRnewp', record = True)
    StateMR0 = StateMonitor(Input_Output_train, 'Rt0', record = True)
    StateMRB = StateMonitor(Input_Output_train, 'B', record = True)
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
    print StateMR.t/ms, len(StateMR.t/ms)
    print StateMR.R_new_p, len(StateMR.R_new_p)

    pRnew, = plot(StateMR.t/ms, StateMR.R_new_p[0], 'b')
    pn, = plot(StateMRS.t/ms, StateMRS.noiseRnewp[0], 'r')
    alpha_p = 3350
    R0_p = 6031.56
    y = (alpha_p / (StateMR0.Rt0[0] - R0_p + alpha_p) + 1)
    pbexp, = plot(StateMR0.t/ms, y, 'g')
    pRt0, = plot(StateMR0.t/ms, StateMR0.Rt0[0], 'k')
    pB, = plot(StateMRB.t/ms, StateMRB.B[0], 'y')
    title('Controlling values for new resistances (useful during potentiation)')
    legend([pn, pbexp, pRnew, pRt0, pB], ["R@t", "R@(t-1)", "variability", "exp base", "B"])
    show()
    

    return weights, timestamps, SM, SpikeMI, SpikeMO, StateMIsyn, SpikeMInh, StateMInh, StateMIsyninh


