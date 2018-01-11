#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:28:16 2017

@author: alpha
"""
import warnings
import inspect
from brian2 import NeuronGroup, Synapses, plot, subplot, zeros, ones, xticks,\
    ylabel, xlabel, xlim, ylim, figure, Group, Subgroup
from collections import OrderedDict

from NCSBrian2Lib.models import neuron_models
from NCSBrian2Lib.models import synapse_models


class NCSGroup(Group):
    """just a bunch of methods that are shared between neurons and connections
    class Group is already used by brian2"""

    def __init__(self):
        self.standaloneVars = []
        self.standaloneParams = OrderedDict()
        self.strParams = {}



    def addStateVariable(self, name, value=None, constant=False, changeInStandalone=True):
        """this method allows you to add a state variable
        (usually defined in equations), that is changeable in standalone mode
        If you pass a value, it will directly set it and decide based on that value,
        if the variable should be shared (scalar) or not (vector)"""
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
            self.__setattr__(name, value)  # TODO: Maybe do that always?

    def setParams(self, params, **kwargs):
        return setParams(self, params, **kwargs)

    def updateParam(self, parname):
        "this is used to update string based params during run (e.g. with gui)"
        for strPar in self.strParams:
            if parname in self.strParams[strPar]:
                self.__setattr__(strPar, self.strParams[strPar])


class Neurons(NeuronGroup, NCSGroup):
    """
    This class is a subclass of NeuronGroup
    You can use it as a NeuronGroup, and everything will be passed to NeuronGroup.
    Alternatively, you can also pass an EquationBuilder object that has all keywords and parameters
    """

    def __init__(self, N, equation_builder=None,
                 params=None,
                 method='euler',
                 num_inputs=3,
                 verbose=False, **Kwargs):

        self.verbose = verbose
        self.num_inputs = num_inputs

        if equation_builder is not None:
            if inspect.isclass(equation_builder):
                self.equation_builder = equation_builder()
            elif isinstance(equation_builder, str):
                equation_builder = getattr(
                    neuron_models, equation_builder)
                self.equation_builder = equation_builder()
            else:
                self.equation_builder = equation_builder
            self.equation_builder.addInputCurrents(num_inputs)
            Kwargs.update(self.equation_builder.keywords)
            if params is not None:
                print(
                    "parameters you provided overwrite parameters from EquationBuilder ")
            else:
                params = self.equation_builder.parameters

        self.initialized = True
        NCSGroup.__init__(self)
        NeuronGroup.__init__(self, N, method=method, **Kwargs)

        self.add_attribute('numSynapses')
        self.numSynapses = 0

        if params is not None:
            setParams(self, params, verbose=verbose)

    def registerSynapse(self):
        self.numSynapses += 1
        if self.verbose:
            print('increasing number of registered Synapses of ' +
                  self.name + ' to ', self.numSynapses)
            print('specified max number of Synapses of ' +
                  self.name + ' is ', self.num_inputs)
        if self.num_inputs < self.numSynapses:
            raise ValueError('There seem so be too many connections to ' +
                             self.name + ', please increase num_inputs')

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


    def __getitem__(self, item):
        """this is from brian2/brian2/groups/neurongroup.py
        """
        if not isinstance(item, slice):
            raise TypeError('Subgroups can only be constructed using slicing syntax')
        start, stop, step = item.indices(self._N)
        if step != 1:
            raise IndexError('Subgroups have to be contiguous')
        if start >= stop:
            raise IndexError('Illegal start/end values for subgroup, %d>=%d' %
                             (start, stop))

        return NCSSubgroup(self, start, stop)


# TODO: find out, if it is possible to have delay as statevariable
class Connections(Synapses, NCSGroup):
    """
    This class is a subclass of Synapses
    You can use it as a Synapses, and everything will be passed to Synapses.
    Alternatively, you can also pass an EquationBuilder object that has all keywords and parameters
    """

    def __init__(self, source, target,
                 equation_builder=None,
                 params=None,
                 method='euler',
                 input_number=None,
                 name='synapses*',
                 verbose=False, **Kwargs):

        NCSGroup.__init__(self)

        self.verbose = verbose
        self.input_number = 0

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
            if verbose:
                print(name, ': target', target.name, 'has',
                      target.numSynapses, 'of', target.num_inputs, 'synapses')
                print('trying to add one more...')
            target.registerSynapse()
            self.input_number = target.numSynapses
            if verbose:
                print('OK!')
                print('input number is: '+ str(self.input_number))
        except ValueError as e:
            raise e
        except AttributeError as e:
            if input_number is not None:
                self.input_number = input_number
            else:
                warnings.warn('you seem to use brian2 NeuronGroups instead of NCSBrian2Lib Neurons for ' +
                              str(target.name) + ', therefore, please specify an input_number yourself')
                raise e

        if equation_builder is not None:
            if inspect.isclass(equation_builder):
                self.equation_builder = equation_builder()
            elif isinstance(equation_builder, str):
                equation_builder = getattr(
                    synapse_models, equation_builder)
                self.equation_builder = equation_builder()
            else:
                self.equation_builder = equation_builder
            self.equation_builder.set_inputnumber(self.input_number)
            Kwargs.update(self.equation_builder.keywords)

            if params is not None:
                self.parameters = params
                print(
                    "parameters you provided overwrite parameters from EquationBuilder ")
            else:
                self.parameters = self.equation_builder.parameters

        try:
            Synapses.__init__(self, source, target=target,
                              method=method,
                              name=name, **Kwargs)
        except Exception as e:
            import sys
            raise type(e)(str(e) + '\n\nCheck Equation for errors!\n' +
                          'e.g. are all units specified correctly at the end \
                          of every line?').with_traceback(sys.exc_info()[2])

    def connect(self, condition=None, i=None, j=None, p=1., n=1,
                skip_if_invalid=False,
                namespace=None, level=0, **Kwargs):
        Synapses.connect(self, condition=condition, i=i, j=j, p=p, n=n,
                         skip_if_invalid=skip_if_invalid,
                         namespace=namespace, level=level + 1, **Kwargs)
        setParams(self, self.parameters, verbose=self.verbose)

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

    def plot(self):
        "simple visualization of synapse connectivity (connected dots and connectivity matrix)"
        S = self
        sourceNeuron = len(S.source)
        targetNeuron = len(S.target)
        figure(figsize=(8, 4))
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


def setParams(briangroup, params, ndargs=None, verbose=False):
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
    if verbose:
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



class NCSSubgroup(Subgroup):
    """this helps to make Subgroups compatible, otherwise the same as Subgroup
    TODO: Some functionality of the package is not compatible with subgroups yet!!!
    """

    def __init__(self, source, start, stop, name=None):
        warnings.warn('Some functionality of this package is not compatible with subgroups yet')
        self.numSynapses = 0 #just initialization to avoid having to initialize it with brian2
        self.num_inputs = 0
        self.registerSynapse = None
        Subgroup.__init__(self, source, start, stop, name)
        self.numSynapses = self.source.numSynapses
        self.num_inputs = self.source.num_inputs
        self.registerSynapse = self.source.registerSynapse #TODO: this is not ideal, as it is not necessary to register a synapse for subgroups!


