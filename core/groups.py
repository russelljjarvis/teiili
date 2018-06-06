#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summary
"""
# @Author: alpren, mmilde
# @Date:   2017-27-07 17:28:16
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2018-06-01 15:38:28
"""
Wrapper class for brian2 Group class.
"""
import warnings
import inspect
from brian2 import NeuronGroup, Synapses, plot, subplot, zeros, ones, xticks,\
    ylabel, xlabel, xlim, ylim, figure, Group, Subgroup
from collections import OrderedDict

from teili.models import neuron_models
from teili.models import synapse_models


class TeiliGroup(Group):
    """just a bunch of methods that are shared between neurons and connections
    class Group is already used by brian2

    Attributes:
        standalone_params (dict): Dictionary of standalone parameters.
        standalone_vars (list): List of standalone variables
        str_params (dict): Name of paramters to be updated
    """

    def __init__(self):
        """Summary
        """
        self.standalone_vars = []
        self.standalone_params = OrderedDict()
        self.str_params = {}

    def add_state_variable(self, name, unit=1, shared=False, constant=False, changeInStandalone=True):
        """this method allows you to add a state variable
        (usually defined in equations), that is changeable in standalone mode
        If you pass a value, it will directly set it and decide based on that value,
        if the variable should be shared (scalar) or not (vector)

        Args:
            name (str): Name of state variable
            unit (int, optional): Unit of respective state variable
            shared (bool, optional): Flag to indicate if state variable is shared
            constant (bool, optional): Flag to indicate if state variable is constant
            changeInStandalone (bool, optional): Flag to indicate if state variable should be subject
                to on-line change in cpp standalone mode.
        """
        if shared:
            size = 1
        else:
            # TODO: Check if this works for neuron as well
            # TODO: Raise error (understandable) if addStateVariable is called
            # before synapses are connected
            size = self.variables['N'].get_value()

        try:
            self.variables.add_array(name, size=size, dimensions=unit.dim,
                                     constant=constant, scalar=shared)
        except AttributeError:  # value.dim will throw an exception, if it has no unit
            self.variables.add_array(name, size=size,
                                     constant=constant, scalar=shared)  # dimensionless

        if changeInStandalone:
            self.standalone_vars += [name]
            # self.__setattr__(name, value)  # TODO: Maybe do that always?

    def set_params(self, params, **kwargs):
        """Summary

        Args:
            params (dict): Key adn value of paramter to be set
            **kwargs: Description

        Returns:
            TYPE: Description
        """
        return set_params(self, params, **kwargs)

    def update_param(self, parname):
        """this is used to update string based params during run (e.g. with gui)

        Args:
            parname (str): Name of paramter to be updated
        """
        for strPar in self.str_params:
            if parname in self.str_params[strPar]:
                self.__setattr__(strPar, self.str_params[strPar])

    def print_equations(self):
        for key, value in sorted(self.equation_builder.keywords.items()):
            print("{} : {}".format(key, value))
        # for key in self.equation_builder.keywords:
        #     print(key, " :")
        #     print(self.equation_builder.keywords[key])

    @property
    def model(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return self.equation_builder.keywords['model']


class Neurons(NeuronGroup, TeiliGroup):
    """
    This class is a subclass of NeuronGroup
    You can use it as a NeuronGroup, and everything will be passed to NeuronGroup.
    Alternatively, you can also pass an EquationBuilder object that has all keywords and parameters

    Attributes:
        equation_builder (TYPE): Class which describes the neuron model equation and all
            porperties and default paramters. See /model/builder/neuron_equation_builder.py and
            models/neuron_models.py
        initialized (bool): Flag to register Neurons population with TeiliGroups
        num_inputs (int): Number of possible synaptic inputs. This overocmes the summed issue
            present in brian2.
        num_synapses (int): Number of synapses projecting to post-synaptic neurn group
        synapses_dict (dict): Dictionary with all synapse names and their respective synapse index
        verbose (bool): Flag to print more details of neurongroup generation
    """

    def __init__(self, N, equation_builder=None,
                 parameters=None,
                 method='euler',
                 # num_inputs=3,
                 verbose=False, **Kwargs):
        """Summary

        Args:
            N (int, required): Number of neurons in respective Neurons groups
            equation_builder (None, optional): Class which describes the neuron model equation and all
                porperties and default paramters. See /model/builder/neuron_equation_builder.py and
                models/neuron_models.py
            params (dict, optional): Dictionary of parameter's keys and values
            method (str, optional): Integration method to solve the differential equation
                present in brian2.
            verbose (bool, optional): Flag to print more details of neurongroup generation
            **Kwargs: Description
        """
        self.verbose = verbose
        #self.num_inputs = num_inputs
        self.num_synapses = 0
        self.synapses_dict = {}

        if equation_builder is not None:
            # if inspect.isclass(equation_builder):
            #    self.equation_builder = equation_builder()
            if isinstance(equation_builder, str):
                equation_builder = getattr(
                    neuron_models, equation_builder)
                self.equation_builder = equation_builder()
            else:
                self.equation_builder = equation_builder
            # self.equation_builder.add_input_currents(num_inputs)
            Kwargs.update(self.equation_builder.keywords)
            Kwargs.pop('parameters')

            if parameters is not None:
                self.parameters = parameters
                print(
                    "parameters you provided overwrite parameters from EquationBuilder ")
            else:
                self.parameters = self.equation_builder.keywords['parameters']

        self.initialized = True
        TeiliGroup.__init__(self)
        NeuronGroup.__init__(self, N, method=method, **Kwargs)

        set_params(self, self.parameters, verbose=verbose)

    def register_synapse(self, synapsename):
        """Summary
        Registers a Synapse so we know the input number.
        It counts all synapses conected with one neurongroup

        Raises:
            ValueError: If too many synapses project to a given post-synaptic neuron groups
                this error is been raised. You need to increae the number of inputs counter

        Args:
            synapsename (str): Name of the synapse group to be registered

        Returns:
            dict: dictionary with all synapse names and their respective synapse index
        """
        if synapsename not in self.synapses_dict:
            self.num_synapses += 1
            self.synapses_dict[synapsename] = self.num_synapses
        if self.verbose:
            print('increasing number of registered Synapses of ' +
                  self.name + ' to ', self.num_synapses)
            # print('specified max number of Synapses of ' +
            #      self.name + ' is ', self.num_inputs)
        # if self.num_inputs < self.num_synapses:
        #    raise ValueError('There seem so be too many connections to ' +
        #                     self.name + ', please increase num_inputs')
        return self.synapses_dict[synapsename]

    def __setattr__(self, key, value):
        """Set attribute method

        Args:
            key (TYPE): key of attribute to be set
            value (TYPE): value of respective key to be set
        """
        NeuronGroup.__setattr__(self, key, value)
        if hasattr(self, 'name'):
            if key in self.standalone_vars and not isinstance(value, str):
                # we have to check if the variable has a value assigned or
                # is assigned a string that is evaluated by brian2 later
                # as in that case we do not want it here
                self.standalone_params.update({self.name + '_' + key: value})

            if isinstance(value, str) and value != 'name' and value != 'when':
                # store this for later update
                self.str_params.update({key: value})

    def __getitem__(self, item):
        """Taken from brian2/brian2/groups/neurongroup.py

        Args:
            item (TYPE): Description

        Returns:
            TeiliSubgroup: The respective neuron subgroup

        Raises:
            IndexError: Error that indicates that size of subgroup set by start and stop is out of bounds
            TypeError: Error to indicate that wrong syntax has been used
        """
        if not isinstance(item, slice):
            raise TypeError(
                'Subgroups can only be constructed using slicing syntax')
        start, stop, step = item.indices(self._N)
        if step != 1:
            raise IndexError('Subgroups have to be contiguous')
        if start >= stop:
            raise IndexError('Illegal start/end values for subgroup, %d>=%d' %
                             (start, stop))

        return TeiliSubgroup(self, start, stop)


# TODO: find out, if it is possible to have delay as statevariable
class Connections(Synapses, TeiliGroup):
    """
    This class is a subclass of Synapses
    You can use it as a Synapses, and everything will be passed to Synapses.
    Alternatively, you can also pass an EquationBuilder object that has all keywords and parameters

    Attributes:
        equation_builder (teili): Class which builds the synapse model
        input_number (int): Number of input to post synatic neuron. This variable takes care of the summed
            issue present in brian2
        parameters (dict): Dictionary of parameter keys and values of the synapse model.
        verbose (bool): Flag to print more detail about synapse generation.
    """

    def __init__(self, source, target,
                 equation_builder=None,
                 parameters=None,
                 method='euler',
                 input_number=None,
                 name='synapses*',
                 verbose=False, **Kwargs):
        """Summary

        Args:
            source (NeuronGroup, Neurons obj.): Pre-synaptic neuron population
            target (NeuronGroup, Neurons obj.): Post-synaptic neuron population
            equation_builder (None, optional): Class which builds the synapse model
            params (dict, optional): Non-default parameter dictionary
            method (str, optional): Integration/Differentiation method used to solve dif. equation
            input_number (int, optional): Number of input to post synatic neuron. This variable takes care of the summed
                issue present in brian2
            name (str, optional): Name of synapse group
            verbose (bool, optional): Flag to print more detail about synapse generation.
            **Kwargs: Addtional keyword arguments.

        Raises:
            e: Description
            type: Description

        No Longer Raises:
            AttributeError: Warning to indicate that an input_number was specified even though this is taken care of automatically
            Exception: Unit mismatch in equations
        """
        TeiliGroup.__init__(self)

        self.verbose = verbose
        self.input_number = 0

        # check if it is a building block, if yes, set bb.group as
        # source/target
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
                      target.num_synapses, 'synapses')
                print('trying to add one more...')
            self.input_number = target.register_synapse(name)
            if self.verbose:
                print('OK!')
                print('input number is: ' + str(self.input_number))

        except ValueError as e:
            raise e
        except AttributeError as e:
            if input_number is not None:
                self.input_number = input_number
            else:
                warnings.warn('you seem to use brian2 NeuronGroups instead of teili Neurons for ' +
                              str(target.name) + ', therefore, please specify an input_number yourself')
                raise e

        if parameters is not None:
            self.parameters = parameters

        if equation_builder is not None:
            if inspect.isclass(equation_builder):
                self.equation_builder = equation_builder()
            elif isinstance(equation_builder, str):
                equation_builder = getattr(
                    synapse_models, equation_builder)
                self.equation_builder = equation_builder()
            else:
                self.equation_builder = equation_builder
            self.equation_builder.set_input_number(self.input_number)
            Kwargs.update(self.equation_builder.keywords)
            Kwargs.pop('parameters')

            if parameters is None:
                self.parameters = self.equation_builder.keywords['parameters']
            else:
                print(
                    "parameters you provided overwrite parameters from EquationBuilder")

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
        """Wrapper function to make synaptic connections among neurongroups

        Args:
            condition (bool, str, optional): A boolean or string expression that evaluates to a boolean. The expression can depend
                on indices i and j and on pre- and post-synaptic variables. Can be combined with arguments n, and p but not i or j.
            i (int, str, optional): Source neuron index
            j (int, str, optional): Target neuron index
            p (float, optional): Probability of connection
            n (int, optional): The number of synapses to create per pre/post connection pair. Defaults to 1.
            skip_if_invalid (bool, optional): Flag to skip connection if invalid indices are given
            namespace (str, optional): namespace of this synaptic connection
            level (int, optional): Description
            **Kwargs: Description
        """
        Synapses.connect(self, condition=condition, i=i, j=j, p=p, n=n,
                         skip_if_invalid=skip_if_invalid,
                         namespace=namespace, level=level + 1, **Kwargs)
        set_params(self, self.parameters, verbose=self.verbose)

    def __setattr__(self, key, value):
        """Function to set arguments to synapses

        Args:
            key (str): Name of attribute to be set
            value (TYPE): Description
        """
        Synapses.__setattr__(self, key, value)
        if hasattr(self, 'name'):
            if key in self.standalone_vars and not isinstance(value, str):
                # we have to check if the variable has a value assigned or
                # is assigned a string that is evaluated by brian2 later
                # as in that case we do not want it here
                self.standalone_params.update({self.name + '_' + key: value})

            if isinstance(value, str) and value != 'name' and value != 'when':
                # store this for later update
                self.str_params.update({key: value})

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


def set_params(briangroup, params, ndargs=None, raise_error=False, verbose=False):
    """This function takes a params dictionary and sets the parameters of a briangroup

    Args:
        brianggroup(brian2.groups.group, required): Neuron or Synapsegroup to set parameters on
        params (dict, required): Parameter keys and values to be set
        raise_error (boolean, optional): determines if an error is raised,
            if a parameter does not exist as a state variable of the group
        ndargs (None, optional): Description
        verbose (bool, optional): Flag to get more details about paramter setting process
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
        else:
            # print and warn, as warnings are sometimes harder to see
            print("Group " + str(briangroup.name) +
                  " has no state variable " + str(par) +
                  ", but you tried to set it with set_params")
            warnings.warn("Group " + str(briangroup.name) +
                          " has no state variable " + str(par) +
                          ", but you tried to set it with set_params")
            if raise_error:
                raise AttributeError("Group " + str(briangroup.name) +
                                     " has no state variable " + str(par) +
                                     ', but you tried to set it with set_params ' +
                                     'if you want to ignore this error, pass raise_error = False')

    if verbose:
        # This fails with synapses coming from SpikeGenerator groups, unidentified bug?
        # This does not work in standalone mode as values of state variables
        # cannot be retrieved before the simulation has been run
        try:
            states = briangroup.get_states()
            print('\n')
            print('-_-_-_-_-_-_-_\nParameters set by set_params for',
                  briangroup.name, ':')
            print('List of first value of each parameter:')
            for key in states.keys():
                if key in params:
                    if states[key].size > 1:
                        print(key, states[key][1])
                    else:
                        print(key, states[key])
            print('----------')

            dellist = list(params.keys()) + \
                ['N', 'i', 't', 'dt', 'not_refractory', 'lastspike']
            for k in dellist:
                try:
                    states.pop(k)
                except:
                    pass
            print('By this set_params call, you have not set the following parameters:')
            for key in states.keys():
                if states[key].size > 1:
                    print(key, states[key][1])
                else:
                    print(key, states[key])

        except:
            print('printing of states does not work in cpp standalone mode')


class TeiliSubgroup(Subgroup):
    """this helps to make Subgroups compatible, otherwise the same as Subgroup
    TODO: Some functionality of the package is not compatible with subgroups yet!!!

    Attributes:
        register_synapse (TYPE): Description
    """

    def __init__(self, source, start, stop, name=None):
        """Summary

        Args:
            source (neurongroup): Neuron group to be split into subgroups
            start (int, required): Start index of source neuron group which should be in subgroup
            stop (int, required): End index of source neuron group which should be in subgroup
            name (str, optional): Name of subgroup
        """
        warnings.warn(
            'Some functionality of this package is not compatible with subgroups yet')
        self.register_synapse = None
        Subgroup.__init__(self, source, start, stop, name)
        # TODO: this is not ideal, as it is not necessary to register a synapse
        # for subgroups!
        self.register_synapse = self.source.register_synapse

    @property
    def num_synapses(self):
        """Property to overcome summed issue present in brian2

        Returns:
            int: Number of synapse which originate at the same pre-synaptic neuron group.
        """
        return self.source.num_synapses

#    @property
#    def num_inputs(self):
#        """Property to overcome summed issue present in brian2
#
#        Returns:
#            int: Number of synapse which project to the same post-synaptic neuron group.
#        """
#        return self.source.num_inputs
