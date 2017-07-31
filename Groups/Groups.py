#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:28:16 2017

@author: alpha
"""
from brian2 import NeuronGroup,Synapses,plot,subplot,zeros,ones,xticks,ylabel,xlabel,xlim,ylim,figure
import warnings

class Neurons(NeuronGroup):
    
    def __init__(self, N, Equation, params,
                 method='euler',
                 refractory=False,
                 events=None,
                 namespace=None,
                 dtype=None,
                 dt=None,
                 clock=None,
                 order=0,
                 name='neurongroup*',
                 codeobj_class=None,
                 #these are additional:
                 numInputs = 1,
                 additionalStatevars = None,
                 debug = False):
        self.debug = debug
        self.__numInputs = numInputs
        self.numSynapses = 0
        # generate the equation in order to pu it into the NeuronGroup constructor
        eqDict = Equation(numInputs = numInputs,debug=debug,method=method, additionalStatevars = additionalStatevars)
        
        NeuronGroup.__init__(self, N, **eqDict,
                 events=events,
                 namespace=namespace,
                 dtype=dtype,
                 dt=dt,
                 clock=clock,
                 order=order,
                 name=name,
                 codeobj_class=codeobj_class)
                
        setParams(self, params, debug = debug)
        self.refP = refractory
          
    def registerSynapse(self):
        self.numSynapses += 1
        if self.debug:
            print('increasing number of registered Synapses of '+ self.name +' to ' , self.numSynapses )
            print('specified max number of Synapses of '+ self.name +' is ' , self.numInputs )
        if self.__numInputs < self.numSynapses:
            raise ValueError ('There seem so be too many connections to '+ self.name + ', please increase numInputs')
        
class Connections(Synapses):
    
    def __init__(self, source, target, Equation, params,
             connect=None, delay=None, on_event='spike',
             multisynaptic_index=None,
             namespace=None, dtype=None,
             codeobj_class=None,
             dt=None, clock=None, order=0,
             method='euler',
             name='synapses*',
             additionalStatevars = None,
             inputNumber = None,
             debug=False):
        
        self.params = params
        self.debug  = debug
        
        try:
            target.registerSynapse()
            self.inputNumber = target.numSynapses
        except ValueError as e:
            raise e
        except:
            if inputNumber is not None:
                self.inputNumber = inputNumber
            else:
                warnings.warn('you seem to use brian2 NeuronGroups instead of NCSBrian2Lib Neurons, therefore, please specify an inputNumber')
        
        synDict = Equation(inputNumber = self.inputNumber , debug=debug, additionalStatevars = additionalStatevars)
        
        Synapses.__init__(self, source, target=target, **synDict,
             connect=connect, delay=delay, on_event=on_event,
             multisynaptic_index=multisynaptic_index,
             namespace=namespace, dtype=dtype,
             codeobj_class=codeobj_class,
             dt=dt, clock=clock, order=order,
             method=method,
             name=name)
        
    def connect(self, condition=None, i=None, j=None, p=1., n=1,
                skip_if_invalid=False,
                namespace=None, level=0):
        Synapses.connect(self, condition=condition, i=i, j=j, p=p, n=n,
                skip_if_invalid=skip_if_invalid,
                namespace=namespace, level=level+1)
        
        setParams(self,self.params,debug=self.debug)

    
    def plot(self):
        "simple visualization of synapse connectivity (connected dots and connectivity matrix)"
        S=self
        Ns = len(S.source)
        Nt = len(S.target)
        fig = figure(figsize=(8, 4))
        subplot(121)
        plot(zeros(Ns), range(Ns), 'ok', ms=10)
        plot(ones(Nt), range(Nt), 'ok', ms=10)
        for i, j in zip(S.i, S.j):
            plot([0, 1], [i, j], '-k')
        xticks([0, 1], ['Source', 'Target'])
        ylabel('Neuron index')
        xlim(-0.1, 1.1)
        ylim(-1, max(Ns, Nt))
        subplot(122)
        plot(S.i, S.j, 'ok')
        xlim(-1, Ns)
        ylim(-1, Nt)
        xlabel('Source neuron index')
        ylabel('Target neuron index')

    
def setParams(briangroup, params, ndargs=None, debug=False):
    for par in params:
        if hasattr(briangroup, par):
            if ndargs is not None and par in ndargs:
                if ndargs[par] is None:
                    setattr(briangroup, par, params[par])
                else:
                    print(par, ndargs, ndargs[par])
                    setattr(briangroup, par, ndargs[par])
            else:
                setattr(briangroup, par, params[par])
    if debug:
        # This fails with synapses coming from SpikeGenerator groups, unidentified bug?
        # This does not work in standalone mode as values of state variables cannot be retrieveed before the simulation has been run
        states = briangroup.get_states()
        print ('\n')
        print ('-_-_-_-_-_-_-_', '\n', 'Parameters set')
        print(briangroup.name)
        print('List of first value of each parameter:')
        for key in states.keys():
            if key in params:
                if states[key].size > 1:
                    print (key, states[key][1])
                else:
                    print (key, states[key])
        print ('----------')
