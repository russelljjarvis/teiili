#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:28:16 2017

@author: alpha
"""
import warnings
from brian2 import NeuronGroup, Synapses, plot, subplot, zeros, ones, xticks,\
    ylabel, xlabel, xlim, ylim, figure
from collections import OrderedDict

# TODO: maybe offer a network argument in order to automatically add the group to the network

# TODO: Change this back to the correct files:
from NCSBrian2Lib.Equations.SynapseEquation import SynapseEquation
from NCSBrian2Lib.Equations.NeuronEquation import NeuronEquation


class Neurons(NeuronGroup):

    def __init__(self, N,
                 params=None,
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
                 # these are additional:
                 numInputs=1,
                 additionalStatevars=None,
                 debug=False, **Kwargs):
        self.debug = debug
        self.numInputs = numInputs
        self.numSynapses = 0


        self.standaloneVars = []
        self.standaloneParams = OrderedDict()
        self.strParams = {}

#        print(self.equation.model) for debugging porpouses

        self.initialized = True
        NeuronGroup.__init__(self, N,
                             events=events,
                             namespace=namespace,
                             dtype=dtype,
                             dt=dt,
                             clock=clock,
                             order=order,
                             name=name,
                             codeobj_class=codeobj_class, method=method, **Kwargs)

        if params is not None:
            setParams(self, params, debug=debug)
        #self.refP = refractory

    def __setattr__(self, key, value):
        NeuronGroup.__setattr__(self, key, value)
        if hasattr(self, 'name'):
            if key in self.standaloneVars and not isinstance(value, str):
                # we have to check if the variable has a value assigned or
                # is assigned a string that is evaluated by brian2 later
                # as in that case we do not want it here
                self.standaloneParams.update({self.name + '_' + key: value})

            if isinstance(value, str) and value != 'name' and value != 'when':
                # store this for later update
                self.strParams.update({key: value})

    def setParams(self, params, **kwargs):
        return setParams(self, params, **kwargs)

    def updateParam(self, parname):
        "this is used to update string based params during run (e.g. with gui)"

        for strPar in self.strParams:
            if parname in self.strParams[strPar]:
                self.__setattr__(strPar, self.strParams[strPar])

    def addStateVariable(self, name, value, constant=False, changeInStandalone=True):
        """this method allows you to add a state variable
        (usually defined in equations), that is changeable in standalone mode"""
        try:
            if len(value) == 1:  # this will probably never happen
                shared = True
                size = 1
            else:
                shared = False
                size = len(value)
                if size != self.N:
                    print('The value of ' + name + ' needs to be a scalar or a vector of\
                          length N (number of neurons in Group)')  # exception will be raised later
        except TypeError:  # then it is probably a scalar
            shared = True
            size = 1

        try:
            self.variables.add_array(name, size=size, dimensions=value.dim,
                                     constant=constant, scalar=shared)
        except AttributeError:  # value.dim will throw an exception, if it has no unit
            self.variables.add_array(name, size=size,
                                     constant=constant, scalar=shared)  # dimensionless

        if changeInStandalone:
            self.standaloneVars += [name]
            self.__setattr__(name, value)

    def registerSynapse(self):
        self.numSynapses += 1
        if self.debug:
            print('increasing number of registered Synapses of ' +
                  self.name + ' to ', self.numSynapses)
            print('specified max number of Synapses of ' +
                  self.name + ' is ', self.numInputs)
        if self.numInputs < self.numSynapses:
            raise ValueError('There seem so be too many connections to ' +
                             self.name + ', please increase numInputs')

    # def print(self):
    #     self.equation.print()


class Connections(Synapses):

    def __init__(self, source, target, 
                 params=None,
                 delay=None, on_event='spike',
                 multisynaptic_index=None,
                 namespace=None, dtype=None,
                 codeobj_class=None,
                 dt=None, clock=None, order=0,
                 method='euler',
                 name='synapses*',
                 additionalStatevars=None,
                 inputNumber=None,
                 debug=False, **Kwargs):

        self.debug = debug
        self.inputNumber = 0

        # check if it is a building block, if yes, set bb.group as source/target
        try:
            target = target.group
        except:
            pass
        try:
            source = source.group
        except:
            pass

        try:
            if debug:
                print(name, ': target', target.name, 'has',
                      target.numSynapses, 'of', target.numInputs, 'synapses')
                print('trying to add one more...')
            target.registerSynapse()
            self.inputNumber = target.numSynapses
        except ValueError as e:
            raise e
        except:
            if inputNumber is not None:
                self.inputNumber = inputNumber
            else:
                warnings.warn('you seem to use brian2 NeuronGroups instead of NCSBrian2Lib Neurons for' +
                              str(target) + ', therefore, please specify an inputNumber')



#        print(self.equation.model) for debugging porpouses

        self.standaloneVars = {}
        self.standaloneParams = OrderedDict()
        self.strParams = {}


        if params is not None:
            setParams(self, params, debug=debug)

        try:
            Synapses.__init__(self, source, target=target,
                              delay=delay, on_event=on_event,
                              multisynaptic_index=multisynaptic_index,
                              namespace=namespace, dtype=dtype,
                              codeobj_class=codeobj_class,
                              dt=dt, clock=clock, order=order,
                              method=method,
                              name=name, **Kwargs)
        except Exception as e:
            import sys
            raise type(e)(str(e) + '\n\nCheck Equation for errors!\n' +
                          'e.g. are all units specified correctly at the end of every line?').with_traceback(sys.exc_info()[2])

    def __setattr__(self, key, value):
        Synapses.__setattr__(self, key, value)
        if hasattr(self, 'name'):
            if key in self.standaloneVars and not isinstance(value, str):
                # we have to check if the variable has a value assigned or
                # is assigned a string that is evaluated by brian2 later
                # as in that case we do not want it here
                self.standaloneParams.update({self.name + '_' + key: value})

            if isinstance(value, str) and value != 'name' and value != 'when':
                # store this for later update
                self.strParams.update({key: value})

    def setParams(self, params, **kwargs):
        return setParams(self, params, **kwargs)

    def updateParam(self, parname):
        "this is used to update string based params during run (e.g. with gui)"

        for strPar in self.strParams:
            if parname in self.strParams[strPar]:
                self.__setattr__(strPar, self.strParams[strPar])

    def addStateVariable(self, name, value, constant=False, changeInStandalone=True):
        """this method allows you to add a state variable
        (usually defined in equations), that is changeable in standalone mode"""
        try:
            if len(value) == 1:  # this will probably never happen
                shared = True
                size = 1
            else:
                shared = False
                size = len(value)
                if size != self.N:
                    print('The value of ' + name + ' needs to be a scalar or a vector of\
                          length N (number of neurons in Group)')  # exception will be raised later
        except TypeError:  # then it is probably a scalar
            shared = True
            size = 1

        try:
            self.variables.add_array(name, size=size, dimensions=value.dim,
                                     constant=constant, scalar=shared)
        except AttributeError:  # value.dim will throw an exception, if it has no unit
            self.variables.add_array(name, size=size,
                                     constant=constant, scalar=shared)  # dimensionless

        if changeInStandalone:
            self.standaloneVars += [name]
            self.__setattr__(name, value)

    def connect(self, condition=None, i=None, j=None, p=1., n=1,
                skip_if_invalid=False,
                namespace=None, level=0):
        Synapses.connect(self, condition=condition, i=i, j=j, p=p, n=n,
                         skip_if_invalid=skip_if_invalid,
                         namespace=namespace, level=level + 1)



    def plot(self):
        "simple visualization of synapse connectivity (connected dots and connectivity matrix)"
        S = self
        sourceNeuron = len(S.source)
        targetNeuron = len(S.target)
        fig = figure(figsize=(8, 4))
        subplot(121)
        plot(zeros(sourceNeuron), range(sourceNeuron), 'ok', ms=10)
        plot(ones(targetNeuron), range(targetNeuron), 'ok', ms=10)
        for i, j in zip(S.i, S.j):
            plot([0, 1], [i, j], '-k')
        xticks([0, 1], [self.source.name, self.target.name])
        ylabel('Neuron index')
        xlim(-0.1, 1.1)
        ylim(-1, max(sourceNeuron, targetNeuron))
        subplot(122)
        plot(S.i, S.j, 'ok')
        xlim(-1, sourceNeuron)
        ylim(-1, targetNeuron)
        xlabel('Source neuron index')
        ylabel('Target neuron index')

    # def print(self):
    #     self.equation.print()


def setParams(briangroup, params, ndargs=None, debug=False):
    """This function takes a params dictionary and sets the parameters of a briangroup"""
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
        # This does not work in standalone mode as values of state variables
        # cannot be retrieved before the simulation has been run
        states = briangroup.get_states()
        print('\n')
        print('-_-_-_-_-_-_-_', '\n', 'Parameters set')
        print(briangroup.name)
        print('List of first value of each parameter:')
        for key in states.keys():
            if key in params:
                if states[key].size > 1:
                    print(key, states[key][1])
                else:
                    print(key, states[key])
        print('----------')
