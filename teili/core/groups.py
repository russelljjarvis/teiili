#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper class for brian2 Group class.

Todo:
    * Check if `shared` works for neuron as well.
    * Raise error (understandable)
        if addStateVariable is called before synapses are connected.
    * Find out, if it is possible to have delay as state variable for Connections.
    * Some functionality of the package is not compatible with subgroups yet.
    * This: `self.register_synapse = self.source.register_synapse` is not ideal,
        as it is not necessary to register a synapse for subgroups!
"""
# @Author: alpren, mmilde
# @Date:   2017-27-07 17:28:16
import os
import importlib
import numpy as np
import warnings
import inspect
from brian2 import NeuronGroup, Synapses, Group, Subgroup, Nameable
from collections import OrderedDict
from matplotlib.pyplot import figure, xlabel, \
    ylabel, plot, subplot, xlim, ylim, xticks
from numpy import ones, zeros

from teili.models import neuron_models
from teili.models import synapse_models
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.tools.random_sampling import Randn_trunc
from teili import constants
from scipy import size
from scipy.stats import truncnorm
from teili.models.parameters.no_mismatch_parameters import no_mismatch_neuron, \
    no_mismatch_synapse


class TeiliGroup(Group):
    """just a bunch of methods that are shared between neurons and
    connections class Group is already used by brian2.

    Attributes:
        standalone_params (dict): Dictionary of standalone parameters.
        standalone_vars (list): List of standalone variables.
        str_params (dict): Name of parameters to be updated.
    """

    def __init__(self):
        """Summary
        """
        self.standalone_vars = []
        self.standalone_params = OrderedDict()
        self.str_params = {}
        self._tags = {}

    def add_state_variable(self, name, unit=1, shared=False, constant=False,
                           changeInStandalone=True):
        """This method allows you to add a state variable.

        Usually a state variable is defined in equations, that is changeable
        in standalone mode. If you pass a value, it will directly set it and
        decide based on that value, if the variable should be shared
        (scalar) or not (vector).

        Args:
            name (str): Name of state variable.
            unit (int, optional): Unit of respective state variable.
            shared (bool, optional): Flag to indicate if state variable is
                shared.
            constant (bool, optional): Flag to indicate if state variable is
                constant.
            changeInStandalone (bool, optional): Flag to indicate if state
                variable should be subject to on-line change in cpp
                standalone mode.
        """
        if shared:
            size = 1
        else:
            size = self.variables['N'].get_value()

        try:
            self.variables.add_array(name, size=size, dimensions=unit.dim,
                                     constant=constant, scalar=shared)
        # value.dim will throw an exception, if it has no unit
        except AttributeError:
            self.variables.add_array(name,
                                     size=size,
                                     constant=constant,
                                     scalar=shared)  # dimensionless

        if changeInStandalone:
            self.standalone_vars += [name]
            # self.__setattr__(name, value)
            # TODO: Maybe do that always?

    def add_subexpression(self, name, dimensions, expr):
        """This method allows you to add a subexpression
        (like a state variable but a string that can be evaluated over time)
        You can e.g. add a timedArray like that:
        >>> neuron_group.add_subexpression('I_arr',nA.dim,'timed_array(t)')
        (be aware, that you need to add a state variable I_arr first,
        that is somehow connected to other variables, so run_regularly
        may be an easier solution for your problem)

        Args:
            name (str): name of the expression.
            dimensions (brian2.units.fundamentalunits.Dimension): dimension
                of the expression.
            expr (str): the expression.
        """
        self.variables.add_subexpression(
            name=name, dimensions=dimensions, expr=expr)

    def set_params(self, params, **kwargs):
        """This function sets parameters on members of a Teiligroup.

        Args:
            params (dict): Key and value of parameter to be set.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The parameters set.
        """
        return set_params(self, params, **kwargs)

    def get_params(self, params=None, verbose=False):
        """This function gets parameter values of neurons or synapses.
        In standalone mode, it only works after the simulation has been run.

        Args:
            params (list, optional): list of parameters that should be
                retrieved. If params = None (default), a dictionary of all
                parameters with their current values is returned

        Returns:
            dict: dictionary of parameters with their values
        """
        states = self.get_states()

        if params is None:
            params = self._init_parameters

        paramdict = {p: states[p] for p in params}

        if verbose:
            print('\n')
            print('Parameters of ' + str(self.name) + ':')
            print_paramdict(paramdict)

        return paramdict

    def update_param(self, parameter_name, verbose=True):
        """This is used to update string based params during run
        (e.g. with gui).

        Args:
            parameter_name (str): Name of parameter to be updated.
        """
        for strPar in self.str_params:
            if parameter_name in self.str_params[strPar]:
                self.__setattr__(strPar, self.str_params[strPar])
                if verbose:
                    print(strPar, 'set to', self.str_params[strPar])

    def print_equations(self):
        """This function print the equation underlying the TeiliGroup
        member.
        """
        print("-_-_-_-_-_-_-_-_")
        for key, value in sorted(self.equation_builder.keywords.items()):
            if type(value) == dict:
                print("Parameters:")
                paramdict = self.get_params()
                print_paramdict(paramdict)
                # for param_key, param_value in sorted(value.items()):
                # print("         {} : {}".format(param_key, param_value))
                print("-_-_-_-_-_-_-_-_")
            else:
                print("{} : {}".format(key, value))
                print("-_-_-_-_-_-_-_-_")

    @property
    def model(self):
        """This property allows the user to only show the model of a given
        member.

        Returns:
            dict: Dictionary only containing model equations.
        """
        return self.equation_builder.keywords['model']

    def add_mismatch(self, std_dict=None, seed=None, verbose=False):
        """
        This function is a wrapper for the method _add_mismatch_param() to
        add mismatch to a dictionary of parameters specified in the input
        dictionary (std_dict). Mismatch is drawn from a Gaussian
        distribution with mean equal to the parameter's current value.

        If no dictionary is given, 20% mismatch is added to all the
        parameters of the model except variables specified in the
        no_mismatch_parameter file.


        Note:
            if you want to specify also lower and upper bound of the
            mismatch distribution see _add_mismatch_param() which adds
            mismatch to a single parameter.

        Args:
            std_dict (dict, optional): dictionary of parameter names as keys
                and standard deviation as values. Standard deviations are
                expressed as fraction of the current parameter value.
                If empty, 20% of missmatch will be added to all variables
                (example: if std_dict = {'Itau': 0.1}, the new parameter
                value will be sampled from a normal distribution with
                standard deviation of 0.1*old_param, with old_param being
                the old parameter value)
            seed (int, optional): seed value for the random generator.
                Set the seed if you want to make the mismatch values
                reproducible across simulations. The random generator state
                before calling this method will be restored after the call
                in order to avoid effects to the rest of your simulation
                (default = None)

        Example:
            Adding mismatch to 100 DPI neurons.

            First create the neuron population (this sets parameter default
            values):
            >>> from teili.models.neuron_models import DPI
            >>> testNeurons = Neurons(100, equation_builder=DPI(num_inputs=2))

            Store the old values as array:
            >>> old_param_value = np.copy(getattr(testNeurons, 'Itau'))

            Add mismatch to the neuron Itau with a standard deviation of
            10% of the current bias values:
            >>> testNeurons.add_mismatch({'Itau': 0.1})
        """
        if std_dict is  None:
            std_dict = {}
            parameters = list(self.equation_builder.keywords['parameters'].keys())
            for i in parameters:
                std_dict[i] = 0.2

        if std_dict is None:
            std_dict = {}
            parameters = list(
                self.equation_builder.keywords['parameters'].keys())

            for i in parameters:
                if i not in no_mismatch_neuron:
                    if i not in no_mismatch_synapse:
                        std_dict[i] = 0.2

        for parameter, std in std_dict.items():
            self._add_mismatch_param(parameter, std, seed=seed)

    def _add_mismatch_param(self, param, std=0, lower=None,
                            upper=None, seed=None):
        """This function sets the input parameter (param) to a value
        (new_param) drawn from a normal distribution with standard deviation
        (std) expressed as a fraction of the current value (old_param).

        Args:
            param (str): name of the parameter to which the mismatch has to
                be added
            std (float): normalized standard deviation, expressed as a
                fraction of the current parameter (e.g.: std = 0.1 means
                that the new value will be sampled from a normal
                distribution with standard deviation of 0.1*old_param, with
                old_value being the current value of the attribute param)
                (default: 0)
            lower (float, optional): lower bound for the parameter mismatch,
                expressed as a fraction of the standard deviation, see note
                below. (default: -1/std)
            upper (float, optional): upper bound for the parameter mismatch,
                expressed as a fraction of the standard deviation, see note
                below. (default: inf)
            seed (int, optional): seed value for the random generator.
                Set the seed if you want to make the mismatch values
                reproducible across simulations. (default: None)

        NOTE: the outuput value (new_param) is drawn from a Gaussian
            distribution with parameters:
            mean:               old_param
            standard deviation: std * old_param
            lower bound:        lower * std * old_param + old_param (default: 0, i.e. lower = -1/std)
            upper bound:        upper * std * old_param + old_param (default: inf)

            using the function truncnorm. For details, see:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html

            The seed will not work in standalone mode so far (TODO)

        Raises:
            NameError: if one of the specified parameters in the disctionary
                is not included in the model.
            AttributeError: if the input parameter to be changed does not
                have units
            UserWarning: if the lower bound is negative
                (i.e. if lower < -1/std) (e.g. if the specified parameter
                is a current, negative values are meaningless)

        Example:
            Adding mismatch to Itau in a population of 100 DPI neurons using
            _add_mismatch_param().
            >>> from teili.models.neuron_models import DPI
            >>> testNeurons = Neurons(100, equation_builder=DPI(num_inputs=2))
            >>> testNeurons._add_mismatch_param(param='Itau', std=0.1)

            This will truncate the distribution at 0, to prevent Itau to
            become negative.

            To specify also the lower bound as 2 times the standard
            deviation:
            >>> testNeurons._add_mismatch_param(param='Ith',
                                                std=0.1,
                                                lower=-2)

        TODO: Consider the mismatch for the parameter 'Cm' as a separate
            case.
        TODO: Add UserWarning if mismatch has been added twice both in numpy
            and standalone mode.
        """

        if hasattr(self, param):
            if seed is not None:
                np_current_state = np.random.get_state()
                np.random.seed(seed)

            if std == 0:
                pass
            else:
                if lower is None:
                    lower = -1 / std
                if lower < -1 / std:
                    warnings.warn(
                        "The output parameter can be negative." +
                        "Set input lower between -1 and 0 to truncate" +
                        "the distribution at 0")

                if upper is None:
                    upper = float('inf')

                randn_trunc = Randn_trunc(lower, upper)
                self.namespace.update({randn_trunc.name: randn_trunc})
                setattr(self,
                        param,
                        param + " * (1 + " + str(std) +
                        ' * '+randn_trunc.name+"())")

            if seed is not None:
                np.random.set_state(np_current_state)
        else:
            raise NameError(
                'Mismatch not added to {} because not included in the' +
                'model parameters'.format(param))

    def import_eq(self, filename):
        """Function to import pre-defined neuron/synapse models.

        Args:
            filename (str): path/to/your/model.py
                Usually synapse models can be found in
                teiliApps/models/equations.

        Returns:
            Dictionary: Dictionary keywords with all relevant
                kwargs, to generate model.

        """
        # if only the filename without path is given, we assume it is one of
        # the predefined models
        fallback_import_path = filename
        if os.path.dirname(filename) is "":
            filename = os.path.join(os.path.expanduser("~"),
                                    'teiliApps',
                                    'models',
                                    'equations',
                                    filename)

        if os.path.basename(filename) is "":
            dict_name = os.path.basename(os.path.dirname(filename))
        else:
            dict_name = os.path.basename(filename)
            filename = os.path.join(filename, '')

        tmp_import_path = []
        while os.path.basename(os.path.dirname(filename)) is not "":
            tmp_import_path.append(os.path.basename(
                os.path.dirname(filename)))
            filename = os.path.dirname(filename)
        importpath = ".".join(tmp_import_path[::-1])

        try:
            eq_dict = importlib.import_module(importpath)
            keywords = eq_dict.__dict__[dict_name]
        except ImportError:
            spec = importlib.util.spec_from_file_location(dict_name[:-3],
                                                          fallback_import_path)
            eq_dict = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eq_dict)
            keywords = eq_dict.__dict__[dict_name[:-3]]

        return keywords


class Neurons(NeuronGroup, TeiliGroup):
    """This class is a subclass of NeuronGroup.

    You can use it as a NeuronGroup, and everything will be passed to
    NeuronGroup. Alternatively, you can also pass an EquationBuilder object
    that has all keywords and parameters.

    Attributes:
        equation_builder (TYPE): Class which describes the neuron model
            equation and all properties and default parameters.
            See /model/builder/neuron_equation_builder.py and
            models/neuron_models.py.
        initialized (bool): Flag to register Neurons' population with
            TeiliGroups.
        num_inputs (int): Number of possible synaptic inputs. This
            overcomes the summed issue present in brian2.
        num_synapses (int): Number of synapses projecting to post-synaptic
            neuron group.
        synapses_dict (dict): Dictionary with all synapse names and their
            respective synapse index.
        verbose (bool): Flag to print more details of neuron group
            generation.
    """

    def __init__(self, N, equation_builder=None,
                 parameters=None,
                 method='euler',
                 verbose=False, **Kwargs):
        """Initializes wrapper for brian2's NeuronGroup class.

        Args:
            N (int, required): Number of neurons in respective Neurons'
                groups.
            equation_builder (None, optional): Class which describes the
                neuron model equation and all porperties and default
                parameters. See /model/builder/neuron_equation_builder.py
                and models/neuron_models.py.
            params (dict, optional): Dictionary of parameter's keys and
                values.
            method (str, optional): Integration method to solve the
                differential equation present in brian2.
            verbose (bool, optional): Flag to print more details of neuron
                group generation.
            **Kwargs: Additional keyword arguments.
        """
        self.verbose = verbose
        self.num_synapses = 0
        self.synapses_dict = {}
        self.parameters = parameters

        if equation_builder is not None:
            # if inspect.isclass(equation_builder):
            #    self.equation_builder = equation_builder()
            if isinstance(equation_builder, dict):
                # if it is a dict, then just take it as it is
                # NeuronEquationBuilder just wraps the dict
                self.equation_builder = NeuronEquationBuilder(
                    keywords=equation_builder)
            elif inspect.isclass(equation_builder):
                self.equation_builder = equation_builder()
            elif isinstance(equation_builder, str):
                self.equation_builder.keywords = import_eq(equation_builder)
            else:
                self.equation_builder = equation_builder
            # self.equation_builder.add_input_currents(num_inputs)

            Kwargs.update(self.equation_builder.keywords)
            Kwargs.update({'method': method})
            Kwargs.pop('parameters')

            if parameters is not None:
                self._init_parameters = parameters
                print(
                    "parameters you provided overwrite parameters from" +
                    "EquationBuilder ")
            else:
                self._init_parameters = self.equation_builder.keywords[
                    'parameters']
        else:
            if parameters is None:
                self._init_parameters = {}
            else:
                self._init_parameters = parameters
        self.initialized = True

        TeiliGroup.__init__(self)
        NeuronGroup.__init__(self, N, **Kwargs)

        set_params(self, self._init_parameters, verbose=verbose)

    def register_synapse(self, synapsename):
        """Registers a Synapse so we know the input number.

        It counts all synapses connected with one neuron group.

        Raises:
            ValueError: If too many synapses project to a given
                post-synaptic neuron group this error is raised.
                You need to increae the number of inputs parameter.

        Args:
            synapsename (str): Name of the synapse group to be registered.

        Returns:
            dict: dictionary with all synapse names and their respective
                 synapse index.
        """
        if synapsename not in self.synapses_dict:
            self.num_synapses += 1
            self.synapses_dict[synapsename] = self.num_synapses
        if self.verbose:
            print('increasing number of registered Synapses of ' +
                  self.name + ' to ', self.num_synapses)

        return self.synapses_dict[synapsename]

    def __setattr__(self, key, value):
        """Set attribute method.

        Args:
            key (TYPE): key of attribute to be set.
            value (TYPE): value of respective key to be set.
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
            TeiliSubgroup: The respective neuron subgroup.

        Raises:
            IndexError: Error that indicates that size of subgroup set by
                start and stop is out of bounds.
            TypeError: Error to indicate that wrong syntax has been used.
        """
        try:
            from brian2.groups.neurongroup import to_start_stop
            start, stop = to_start_stop(item, self._N)
        except ImportError:
            start, stop, step = item.indices(self._N)

        return TeiliSubgroup(self, start, stop)


class Connections(Synapses, TeiliGroup, Nameable):
    """This class is a subclass of Synapses.

    You can use it as a Synapses, and everything will be passed to Synapses.
    Alternatively, you can also pass an EquationBuilder object that has all
    keywords and parameters.

    Attributes:
        equation_builder (teili): Class which builds the synapse model.
        input_number (int): Number of input to post synaptic neuron. This
            variable takes care of the summed issue present in brian2.
        parameters (dict): Dictionary of parameter keys and values of the
            synapse model.
        verbose (bool): Flag to print more detail about synapse generation.
    """

    def __init__(self, source, target,
                 equation_builder=None,
                 parameters=None,
                 method='euler',
                 input_number=None,
                 name='synapses*',
                 verbose=False, **Kwargs):
        """Initializes wrapper for brian2's Synapses class.

        Args:
            source (NeuronGroup, Neurons obj.): Pre-synaptic neuron
                population.
            target (NeuronGroup, Neurons obj.): Post-synaptic neuron
                population.
            equation_builder (None, optional): Class which builds the
                synapse model.
            params (dict, optional): Non-default parameter dictionary.
            method (str, optional): Integration/Differentiation method
                used to solve differential equation.
            input_number (int, optional): Number of input to post synaptic
                neuron. This variable takes care of the summed issue present
                in brian2.
            name (str, optional): Name of synapse group.
            verbose (bool, optional): Flag to print more detail about
                synapse generation.
            **Kwargs: Additional keyword arguments.

        Raises:
            AttributeError e: Warning to indicate that an input_number was
            specified even though this is taken care of automatically.
            type: Unit mismatch in equations.
        """
        TeiliGroup.__init__(self)

        self.verbose = verbose
        self.input_number = 0

        # check if it is a building block, if yes, set bb.input_groups or output_groups as
        # source/target
        try:
            if len(target.input_groups) > 1:
                print(
                    'the building block you are connecting has more than one input group, an arbitrary one is selected')
            target = list(target.input_groups.values())[0]
        except AttributeError:
            pass
        except IndexError as e:
            print('the building block you are trying to connect does not have a valid input group, please select one manually')
            raise(e)
        try:
            if len(target.output_groups) > 1:
                print(
                    'the building block you are connecting has more than one output group, an arbitrary one is selected')
            source = list(source.output_groups.values())[0]
        except AttributeError:
            pass
        except IndexError as e:
            print('the building block you are trying to connect does not have a valid output group, please select one manually')
            raise(e)

        Nameable.__init__(self, name=name)

        try:
            if self.verbose:
                print(self.name, ': target', target.name, 'has',
                      target.num_synapses, 'synapses')
                print('trying to add one more...')
            self.input_number = target.register_synapse(self.name)
            if self.verbose:
                print('OK!')
                print('input number is: ' + str(self.input_number))
        except ValueError as e:
            raise e
        except AttributeError as e:
            if input_number is not None:
                self.input_number = input_number
            else:
                warnings.warn('you seem to be using brian2 NeuronGroups' +
                              'instead of teili Neurons for ' +
                              str(target.name) + ', therefore, please' +
                              'specify an input_number yourself')
                #raise e
        except KeyError as e:
            if input_number is not None:
                self.input_number = input_number
            else:
                warnings.warn('you seem to be using brian2 NeuronGroups' +
                              'instead of teili Neurons for ' +
                              str(target.name) + ', therefore, please' +
                              'specify an input_number yourself')
                #raise e

        if parameters is not None:
            self._init_parameters = parameters

        if equation_builder is not None:
            if isinstance(equation_builder, dict):
                # if it is a dict, then just take it as it is
                # NeuronEquationBuilder just wraps the dict
                self.equation_builder = SynapseEquationBuilder(
                    keywords=equation_builder)
            elif inspect.isclass(equation_builder):
                self.equation_builder = equation_builder()
            elif isinstance(equation_builder, str):
                self.equation_builder.keywords = import_eq(equation_builder)
            else:
                # this copies the object using the call,
                self.equation_builder = equation_builder()
                # it is convenient for the user, but maybe too confusing

            self.equation_builder.set_input_number(self.input_number - 1)
            Kwargs.update(self.equation_builder.keywords)
            Kwargs.pop('parameters')

            if parameters is None:
                self._init_parameters = self.equation_builder.keywords[
                    'parameters']
            else:
                print(
                    "parameters you provided overwrite parameters from EquationBuilder")

        try:
            Synapses.__init__(self, source, target=target,
                              method=method,
                              name=self.name, **Kwargs)
        except Exception as e:
            import sys
            raise type(e)(str(e) + '\n\nCheck Equation for errors!\n' +
                          'e.g. are all units specified correctly at the end ' +
                          'of every line?\n' +
                          'e.g. is the value of num_inputs correct?\n' +
                          'e.g. are you connecting the correct populations?'\
                          ).with_traceback(sys.exc_info()[2])

    def connect(self, condition=None, i=None, j=None, p=1., n=1,
                skip_if_invalid=False,
                namespace=None, level=0, **Kwargs):
        """Wrapper function to make synaptic connections among neurongroups.

        Args:
            condition (bool, str, optional): A boolean or string expression
                that evaluates to a boolean. The expression can depend on
                indices i and j and on pre- and post-synaptic variables.
                Can be combined with arguments n, and p but not i or j.
            i (int, str, optional): Source neuron index.
            j (int, str, optional): Target neuron index.
            p (float, optional): Probability of connection.
            n (int, optional): The number of synapses to create per pre/post
                connection pair. Defaults to 1.
            skip_if_invalid (bool, optional): Flag to skip connection if
                invalid indices are given.
            namespace (str, optional): namespace of this synaptic
                connection.
            level (int, optional): Distance to the input layer needed for
                tags.
            **Kwargs: Additional keyword arguments.
        """
        Synapses.connect(self, condition=condition, i=i, j=j, p=p, n=n,
                         skip_if_invalid=skip_if_invalid,
                         namespace=namespace, level=level + 1, **Kwargs)
        set_params(self, self._init_parameters, verbose=self.verbose)

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
        """Simple visualization of synapse connectivity (connected dots and
        connectivity matrix)
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
    """This function takes a params dictionary and sets the parameters of
    a briangroup.

    Args:
        brianggroup(brian2.groups.group, required): Neuron or Synapsegroup
            to set parameters on.
        params (dict, required): Parameter keys and values to be set.
        raise_error (boolean, optional): determines if an error is raised
            if a parameter does not exist as a state variable of the group.
        ndargs (dict, optional): Addtional attribute arguments.
        verbose (bool, optional): Flag to get more details about parameter
            setting process. States are not printed in cpp standalone mode
            before the simulation has been run
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
                                     ', but you tried to set it with' +
                                     'set_params if you want to ignore this' +
                                     'error, pass raise_error = False')

    if verbose:
        # This fails with synapses coming from SpikeGenerator groups,
        # unidentified bug?
        # This does not work in standalone mode as values of state variables
        # cannot be retrieved before the simulation has been run
        try:
            states = briangroup.get_states()
            print('\n')
            print('-_-_-_-_-_-_-_\nParameters set by set_params for',
                  briangroup.name, ':')
            paramdict = {p: states[p] for p in params}
            print_paramdict(paramdict)
            print('----------')
            dellist = list(params.keys()) + \
                ['N', 'i', 't', 'dt', 'not_refractory', 'lastspike']
            for k in dellist:
                try:
                    states.pop(k)
                except:
                    pass
            print('In this set_params call, you have not set the following' +
                  'parameters:')
            print_paramdict(states)
        except:
            print('Printing of states does not work in cpp standalone mode')


def print_paramdict(paramdict):
    """This function prints a params dictionary for get and set_params.

    Args:
        paramdict (dict, required): Parameter keys and values to be set.
    """
    print('Printing the first value of each parameter:')
    for key in paramdict.keys():
        if paramdict[key].size > 1:
            print(key, '=', paramdict[key][1])
        else:
            print(key, '=', paramdict[key])


class TeiliSubgroup(Subgroup):
    """This helps to make Subgroups compatible, otherwise the same as
    Subgroup.

    Attributes:
        register_synapse (fct): Register a synapse group to TeiliGroup.
    """

    def __init__(self, source, start, stop, name=None):
        """Summary

        Args:
            source (neurongroup): Neuron group to be split into subgroups.
            start (int, required): Start index of source neuron group which
                should be in subgroup.
            stop (int, required): End index of source neuron group which
                should be in subgroup.
            name (str, optional): Name of subgroup.
        """
        warnings.warn(
            'Some functionality of this package is not compatible with' +
            'subgroups yet')
        self.register_synapse = None
        Subgroup.__init__(self, source, start, stop, name)

        self.register_synapse = self.source.register_synapse

    @property
    def num_synapses(self):
        """Property to overcome summed issue present in brian2.

        Returns:
            int: Number of synapses that converge to the same
                post-synaptic neuron group.
        """
        return self.source.num_synapses
