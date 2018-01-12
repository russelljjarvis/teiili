#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: alpren, mmilde
# @Date:   2017-27-07 17:28:16
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-12 11:41:48
"""
Summary
"""
import warnings
import inspect
from brian2 import NeuronGroup, Synapses, plot, subplot, zeros, ones, xticks,\
    ylabel, xlabel, xlim, ylim, figure, Group
from collections import OrderedDict

from NCSBrian2Lib.models import neuron_models
from NCSBrian2Lib.models import synapse_models


class NCSGroup(Group):
    """just a bunch of methods that are shared between neurons and connections
    class Group is already used by brian2

    Attributes:
        standaloneParams (TYPE): Description
        standaloneVars (list): Description
        strParams (dict): Description
    """

    def __init__(self):
        """Summary
        """
        self.standaloneVars = []
        self.standaloneParams = OrderedDict()
        self.strParams = {}

    def addStateVariable(self, name, unit=1, shared=False, constant=False, changeInStandalone=True):
        """this method allows you to add a state variable
        (usually defined in equations), that is changeable in standalone mode
        If you pass a value, it will directly set it and decide based on that value,
        if the variable should be shared (scalar) or not (vector)

        Args:
            name (TYPE): Description
            unit (int, optional): Description
            shared (bool, optional): Description
            constant (bool, optional): Description
            changeInStandalone (bool, optional): Description
        """
        if shared:
            size = 1
        else:
            # TODO: Check if this works for neuron as well
            # TODO: Raise error (understandable) if addStateVariable is called before synapses are connected
            size = self.variables['N'].get_value()

        try:
            self.variables.add_array(name, size=size, dimensions=unit.dim,
                                     constant=constant, scalar=shared)
        except AttributeError:  # value.dim will throw an exception, if it has no unit
            self.variables.add_array(name, size=size,
                                     constant=constant, scalar=shared)  # dimensionless

        if changeInStandalone:
            self.standaloneVars += [name]
            # self.__setattr__(name, value)  # TODO: Maybe do that always?

    def setParams(self, params, **kwargs):
        """Summary

        Args:
            params (TYPE): Description
            **kwargs: Description

        Returns:
            TYPE: Description
        """
        return setParams(self, params, **kwargs)

    def updateParam(self, parname):
        """this is used to update string based params during run (e.g. with gui)

        Args:
            parname (TYPE): Description
        """
        for strPar in self.strParams:
            if parname in self.strParams[strPar]:
                self.__setattr__(strPar, self.strParams[strPar])


class Neurons(NeuronGroup, NCSGroup):
    """
    This class is a subclass of NeuronGroup
    You can use it as a NeuronGroup, and everything will be passed to NeuronGroup.
    Alternatively, you can also pass an EquationBuilder object that has all keywords and parameters

    Attributes:
        equation_builder (TYPE): Description
        initialized (bool): Description
        num_inputs (TYPE): Description
        numSynapses (int): Description
        verbose (TYPE): Description
    """

    def __init__(self, N, equation_builder=None,
                 params=None,
                 method='euler',
                 num_inputs=3,
                 verbose=False, **Kwargs):
        """Summary

        Args:
            N (TYPE): Description
            equation_builder (None, optional): Description
            params (None, optional): Description
            method (str, optional): Description
            num_inputs (int, optional): Description
            verbose (bool, optional): Description
            **Kwargs: Description
        """
        self.verbose = verbose
        self.num_inputs = num_inputs
        self.numSynapses = 0



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

        if params is not None:
            setParams(self, params, verbose=verbose)

    def registerSynapse(self):
        """Summary

        Raises:
            ValueError: Description
        """
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
        """Summary

        Args:
            key (TYPE): Description
            value (TYPE): Description
        """
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


# TODO: find out, if it is possible to have delay as statevariable
class Connections(Synapses, NCSGroup):
    """
    This class is a subclass of Synapses
    You can use it as a Synapses, and everything will be passed to Synapses.
    Alternatively, you can also pass an EquationBuilder object that has all keywords and parameters

    Attributes:
        equation_builder (TYPE): Description
        input_number (int): Description
        parameters (TYPE): Description
        verbose (TYPE): Description
    """

    def __init__(self, source, target,
                 equation_builder=None,
                 params=None,
                 method='euler',
                 input_number=None,
                 name='synapses*',
                 verbose=False, **Kwargs):
        """Summary

        Args:
            source (TYPE): Description
            target (TYPE): Description
            equation_builder (None, optional): Description
            params (None, optional): Description
            method (str, optional): Description
            input_number (None, optional): Description
            name (str, optional): Description
            verbose (bool, optional): Description
            **Kwargs: Description

        Raises:
            e: Description
            type: Description
        """
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
            if self.verbose:
                print(name, ': target', target.name, 'has',
                      target.numSynapses, 'of', target.num_inputs, 'synapses')
                print('trying to add one more...')
            target.registerSynapse()
            self.input_number = target.numSynapses
            if self.verbose:
                print('OK!')
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
        """Summary

        Args:
            condition (None, optional): Description
            i (None, optional): Description
            j (None, optional): Description
            p (float, optional): Description
            n (int, optional): Description
            skip_if_invalid (bool, optional): Description
            namespace (None, optional): Description
            level (int, optional): Description
            **Kwargs: Description
        """
        Synapses.connect(self, condition=condition, i=i, j=j, p=p, n=n,
                         skip_if_invalid=skip_if_invalid,
                         namespace=namespace, level=level + 1, **Kwargs)
        setParams(self, self.parameters, verbose=self.verbose)

    def __setattr__(self, key, value):
        """Summary

        Args:
            key (TYPE): Description
            value (TYPE): Description
        """
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
        """simple visualization of synapse connectivity (connected dots and connectivity matrix)
        """
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
    """This function takes a params dictionary and sets the parameters of a briangroup

    Args:
        briangroup (TYPE): Description
        params (TYPE): Description
        ndargs (None, optional): Description
        verbose (bool, optional): Description
    """
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
