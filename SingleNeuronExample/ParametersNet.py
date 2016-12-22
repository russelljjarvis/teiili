# -*- coding: utf-8 -*-

from brian2 import *
import numpy as np

#-----------------------------------------------------
#PARAMETERS
#-----------------------------------------------------
N_in = 1
num_examples = 5  #numero img training
training_img_time = 0.501 * second #Time of single image during training
testing_img_time = 0.351 * second   #Time of a single image during testing
resting_time = 0.05 * second #ogni volta che presento un training resetto tutti i valori dell'ingresso del neurone
output_classes = 1   # output classes
neurons_per_class = 1 # number of output neuron per class
N_out = output_classes * neurons_per_class  #total number of output neurons
N_inh = 1
tauexc = 5 * msecond #Excitatory teacher time constant 
tauinhib = 5 * msecond
tauarp= 0.01 * msecond       # Absolute refractory period
#--------------------------------------------------------
# VLSI process parameters
#--------------------------------------------------------
kn = 0.75
kp = 0.66
kappa = (kn + kp) / 2
Ut = 25 * mV
Io = 0.5 * pA
#
# Silicon neuron parameters                      
Csyn = 0.1 * pF
Cmem = 0.5 * pF
Cahp = 0.5 * pF
#
# Fitting parameters
Iagain = 1 * nA
Iath   = 20 * nA
Ianorm = 1 * nA
#---------------------------------------------------------
#Adaptative and Calcium parameters
#---------------------------------------------------------
tauahp = 10 * msecond #Adaptive spike decay rate
tauca = 40 * msecond #Calcium spike decay rate
Iposa = 0.3 * pA
Iwa   = 0 * pA #Adaptation spike amplitude
Itaua = 1 * pA
#---------------------------------------------------------
# Neuron parameters
#---------------------------------------------------------
Ispkthr = 20 * nA #Spike threshold MODIFICABILE
Ispkthr_inh = 5 * nA #Spike threshold MODIFICABILE
Ireset = 1 * pA #Reset Imem to Ireset after each spike 
Ith = 1 * pA
Itau = 1 * pA
Imemthr = 7.0 * nA #Minimum Imem for (+) weight change MODIFICABILE 
taum = Cmem * Ut / (kappa * Itau)
tauahp = Cahp * Ut / (kappa * Itaua)
#----------------------------------------------------------
# Synapse parameters
#----------------------------------------------------------
Iw = 0.5 * nA #5.5 * (float(32) / float(N_in)) * nA #Plastic synaptic efficacy CORRENTE ASSOCIATA AGLI SPIKE, MODIFICABILE
tausyn = 5 * ms
Iw_in_h = 0.5 * nA #5.5 * (float(32) / float(N_in)) * nA #Plastic synaptic efficacy DEI N INIBITORI -> INHIBITORY PIÙ EFFICACI SE i ALTA
Iwinh = - 0.5 * nA #SECONDO STADIO DELLA PARTE INIBITORIA POI VA A SOTTRARSI AI NEURONI DI USCITA, MENTRE SOPRA È IN INGRESSO INSIEME ALL'INPUT
Iwexc = 0.5 * nA
