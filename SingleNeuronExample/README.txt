README: SingleNeuronsExample
Author: Daniele Conti
mail: daniele.conti@polito.it
date: 22/12/2016

Network structure:

                                           (1 teacher neuron)
                                                   |
                                         (1 non plastic syn_C)
                                                   |
                                                   V
    (1 input neuron)--->(1 memfusi syn)--->(1 output neuron)
           |                                       A
           |                                       |
           V                                       |
 (1 non plastic syn_A)--->(1 inhib neu)--->(1 non plastic syn_B)


Run the network from main.py
Input to the network: Poisson neuron spiking with frequency 100Hz

Total input current to the output neuron is composed as follow:
Iin = Iin_ex + Iin_inh + Iin_teach
Iin_ex = current from (1 memfusi syn) ---> (1 output neuron)
Iin_inh (negative) = current from (1 non plastic syn_B) ---> (1 output neuron)
Iin_ex = current from (1 non plastic syn_C) ---> (1 output neuron)

Memristive Fusi synapse model is taken from synapseEquations.py (copied into folder)
All synapses models and neurons equations not already present in NCSBrian2Lib are taken from NeuronsSynapses.py
Parameters for all equations but Memristive Fusi synapses are also specified in Training.py, imported from ParametersNet.py
Memristive Fusi Synapse Parameters are read from memfusi_parameters.py

The network is designed to potentiate memfusi-syn whenever teacher is on (300 Hz) and depress memfusi-syn when teacher is off (set rates=0*Hz in NeuronGroup definition for teacher in main.py)



WARNING:
It happens that Imem goes toward overflow if teacher is ON = 300*Hz and if memristive update account for variability. 
There seems to be a strong correlation between noise on memristive update and Imem overflow (50% cases on 10 runs), that disappears if noise is off (0% cases on 10 runs).
Reasons and further correlations are still to be investigated...
The present example is reported with noise on memristive update commented (see lines 87 and 88 in synapseEquations.py) 

