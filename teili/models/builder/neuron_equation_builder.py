# -*- coding: utf-8 -*-
"""This file contains a class that manages a neuron equation.

And it prepares a dictionary of keywords for easy synapse creation.
It also provides a function to add lines to the model.

Example:
    To use the NeuronEquationBuilder "on the fly"
    you can initialize it as DPI neuron:

    >>> from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
    >>> num_inputs = 2
    >>> my_neu_model = NeuronEquationBuilder.__init__(base_unit='current', adaptation='calcium_feedback',
                                       integration_mode='exponential', leak='leaky',
                                       position='spatial', noise='none')
    >>> my_neuron.add_input_currents(num_inputs)

    Or if you have a pre-defined neuron model you can import this dictionary as:

    >>> from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
    >>> my_neu_model = NeuronEquationBuilder.import_eq(
        '~/teiliApps/equations/DPI', num_inputs=2)

    in both cases you can pass it to Neurons:

    >>> from teili.core.groups import Neurons
    >>> my_neuron = Neurons(2, equation_builder=my_neu_model, name="my_neuron")

    Another way of using it is to import the DPI class directly:

    >>> from teili.models.neuron_models import DPI
    >>> from teili.core.groups import Neurons
    >>> my_neuron = Neurons(2, equation_builder=DPI(num_inputs=2), name="my_neuron")
"""
# @Author: mrax, alpren, mmilde
# @Date:   2018-01-12 11:34:34

import os
import importlib
import re
import copy
import warnings
from brian2 import pF, nS, mV, ms, pA, nA
from teili.models.builder.combine import combine_neu_dict
from teili.models.builder.templates.neuron_templates import modes, current_equation_sets, \
    voltage_equation_sets, \
    current_parameters, voltage_parameters


class NeuronEquationBuilder():
    """Class which builds neuron equation according to pre-defined properties such
    as spike-frequency adaptation, leakage etc.

    Attributes:
        keywords (dict): Dictionary containing all relevant keywords and equations for
            brian2, such as model, refractory, reset, threshold and parameters.
        num_inputs (int): Number specifying how many distinct neuron population project
            to the target neuron population.
        verbose (bool): Flag to print more detailed output of neuron equation builder.
    """

    def __init__(self, keywords=None, base_unit='current', num_inputs=1, verbose=False, **kwargs):

        """Initializes NeuronEquationBuilder with defined keyword arguments.

        Args:
            keywords (dict, optional): Brian2 like model.
            base_unit (str, optional): Indicates if neuron is current-based or conductance-based.
            num_inputs (int, optional): Number specifying how many distinct neuron population
                project to the target neuron population.
            verbose (bool, optional): Flag to print more detailed output of neuron equation builder.
            **kwargs (str, optional): dictionary of equations such as:
                 adaptation (str, optional): What type of adaptive feedback should be used.
                     So far only calciumFeedback is implemented.
                integration_mode (str, optional): Sets if integration up to spike-generation is
                   linear or exponential.
                leak (str, optional): Enables leaky integration.
                position (str, optional): To enable spatial-like position indices on neuron.
                noise (str, optional): NOT YET IMPLMENTED! This will in the future allow independent
                mismatch-like noise to be added on each neuron.
                refractory (str, optional): Refractory period of the neuron.
        """
        self.verbose = verbose
        if keywords is not None:
            self.keywords = {'model': keywords['model'],
                             'threshold': keywords['threshold'],
                             'reset': keywords['reset'],
                             'refractory': 'refP',
                             'parameters': keywords['parameters']}

        else:
            ERRValue = """
                                ---Model not present in dictionaries---
                    This class constructor builds a model for a neuron using pre-existing blocks.

                    The first entry is the model type,
                    choose between 'current' or 'voltage'.

                    You can choose then what module to load for your neuron,
                    the entries are 'adaptation', 'exponential', 'leaky', 'spatial', 'gaussian'.
                    If you don't want to load a module just use the keyword 'none'
                    example: NeuronEquationBuilder('current','none','exponential','leak','none','none'.....)

                    """

            try:
                modes[base_unit]
                for key, value in kwargs.items():
                    current_equation_sets[value]

            except KeyError as e:
                print(ERRValue)

            if base_unit == 'current':
                eq_templ_dummy = []
                for key, value in kwargs.items():
                    eq_templ_dummy = eq_templ_dummy + \
                        [current_equation_sets[value]]
                eq_templ = [modes[base_unit]] + eq_templ_dummy

                param_templ_dummy = []
                for key, value in kwargs.items():
                    param_templ_dummy = param_templ_dummy + \
                        [current_parameters[value]]
                param_templ = [current_parameters[base_unit]] + \
                    param_templ_dummy

                if self.verbose:
                    print("Equations", eq_templ)
                    print("Parameters", eq_templ)

                keywords = combine_neu_dict(eq_templ, param_templ)

            if base_unit == 'voltage':
                eq_templ_dummy = []
                for key, value in kwargs.items():
                    eq_templ_dummy = eq_templ_dummy + \
                        [voltage_equation_sets[value]]
                eq_templ = [modes[base_unit]] + eq_templ_dummy
                param_templ_dummy = []
                for key, value in kwargs.items():
                    param_templ_dummy = param_templ_dummy + \
                        [voltage_parameters[value]]
                param_templ = [voltage_parameters[base_unit]] + \
                    param_templ_dummy

                if self.verbose:
                    print("Equations", eq_templ)
                    print("Parameters", eq_templ)

                keywords = combine_neu_dict(eq_templ, param_templ)

            self.keywords = {'model': keywords['model'],
                             'threshold': keywords['threshold'],
                             'reset': keywords['reset'],
                             'refractory': 'refP',
                             'parameters': keywords['parameters']}

            self.num_inputs = num_inputs
            self.add_input_currents(num_inputs)

        if self.verbose:
            self.print_all()

    def __call__(self, num_inputs):
        """In the recommended way of using Neurons as provided bey teili
        the neuron model is imported from teili.models.neuron_models as
        properly initialised python object in which the number of incoming
        current, i.e. num_inputs, is set during the initialisation of the
        class. However, teili also supports to initialise the `Equation_builder`
        using a user-specified model without the need to implement the model
        directly in the existing software stack. This allows faster development
        time and mre flexibility as all functionality of teili is provided to
        user-specified models. This function allows the user to set the
        num_inputs argument to non-standard neuron model.

        An usage example can be found in
        `teiliApps/tutorials/neuron_synapse_builderobj_tutorial.py`
        Args:
            num_inputs (int, required): Number specifying how many distinct
                neuron populations project to the target neuron population.

        Returns:
            NeuronEquationBuilder obj.: A deep copy of the NeuronEquationBuilder object.
        """
        builder_copy = copy.deepcopy(self)
        builder_copy.add_input_currents(num_inputs)
        return builder_copy

    def add_input_currents(self, num_inputs):
        """Automatically adds the input current line according to num_inputs.

        It also adds all these input currents as state variables.

        Example:
            >>> Iin = Ie0 + Ii0 + Ie1 + Ii1 + ... + IeN + IiN (with N = num_inputs)

        Args:
            num_inputs (int): Number of inputs to the post-synaptic neuron
        """
        self.num_inputs = num_inputs

        if num_inputs > 10:
            warnings.warn(
                '''num_inputs of this Neuron is larger than 10.
                Too large values may cause parser problems,
                please check the documentation if you are using num_inputs correctly
                (only different groups need different inputs)''')

        # remove previously added inputcurrent lines
        inputcurrent_pattern = re.compile("Iin\d+ : amp")
        model = self.keywords['model'].split('\n')
        for line in self.keywords['model'].split('\n'):
            if "Iin =" in line or "Iin=" in line:
                model.remove(line)
                if self.verbose:
                    print(
                        'previously added input currents were removed, following lines deleted:')
                    print(line)
            elif inputcurrent_pattern.search(line) is not None:
                if self.verbose:
                    print(line)
                model.remove(line)

        self.keywords['model'] = '\n'.join(model)

        Iins = ["Iin0 "] + ["+ Iin" +
                            str(i + 1) + " " for i in range(num_inputs - 1)]

        self.keywords['model'] = self.keywords['model'] + "\n         Iin = " + \
            "".join(Iins) + " : amp # input currents\n\n"
        Iinsline = ["         Iin" +
                    str(i) + " : amp" for i in range(num_inputs)]
        self.add_state_vars(Iinsline)
        self.keywords['model'] += "\n"

    def add_state_vars(self, stateVars):
        """this function adds state variables to neuron equation by just adding
        a line to the neuron model equation.

        Args:
            stateVars (dict): State variable to be added to neuron model
        """
        if self.verbose:
            print("added to Equation: \n" + "\n".join(stateVars))
        self.keywords['model'] += "\n".join(stateVars)

    def print_all(self):
        """Method to print all dictionaries within a neuron model
        """
        print('Model equation:')
        print(self.keywords['model'])
        print('-_-_-_-_-_-_-_-')
        print('Threshold equation:')
        print(self.keywords['threshold'])
        print('-_-_-_-_-_-_-_-')
        print('Reset equation:')
        print(self.keywords['reset'])
        print('-_-_-_-_-_-_-_-')
        print('Parameters:')
        print_param_dictionaries(self.keywords['parameters'])
        print('-_-_-_-_-_-_-_-')

    def export_eq(self, filename):
        """Function to export generated neuron model to a file.

        Args:
            filename (str): Path/to/location/to/store/neuron_model.py.
        """
        with open(filename + ".py", 'w') as file:
            file.write('from brian2.units import * \n')
            file.write(os.path.basename(filename) + " = {")
            file.write("'model':\n")
            file.write("'''")
            file.write(self.keywords['model'])
            file.write("''',\n")
            file.write("'threshold':\n")
            file.write("'''")
            file.write(self.keywords['threshold'])
            file.write("''',\n")
            file.write("'reset':\n")
            file.write("'''")
            file.write(self.keywords['reset'])
            file.write("''',\n")
            file.write("'parameters':\n")
            file.write("{\n")
            for keys, values in self.keywords['parameters'].items():
                if type(values) is str:
                    writestr = "'" + keys + "'" + " : " + repr(values)
                else:
                    writestr = "'" + keys + "'" + " : '" + repr(values) + "'"
                file.write(writestr)
                file.write(",\n")
            file.write("}\n")
            file.write("}")

    @classmethod
    def import_eq(cls, filename, num_inputs=1):
        """Function to import pre-defined neuron_model.

        num_inputs is used to add additional input currents that are used
        for different synapses that are summed.

        Args:
            filename (str): Path/to/location/where/model/is/stored/neuron_model.py.
            num_inputs (int): Number of inputs to the post-synaptic neuron.

        Returns:
            NeuronEquationBuilder obj.: Object containing keywords (dict) with all relevant
                keys, to generate neuron_model.
        """
        # if only the filename without path is given, we assume it is one of
        # the predefined models
        fallback_import_path = filename
        if os.path.dirname(filename) is "":
            filename = os.path.join(os.path.expanduser('~'),
                                    "teiliApps",
                                    "equations",
                                    filename)
            if not os.path.basename(filename)[-3:] == ".py":
                filename += ".py"
            fallback_import_path = filename

        if os.path.basename(filename) is "":
            dict_name = os.path.basename(os.path.dirname(filename))
        else:
            dict_name = os.path.basename(filename)
            filename = os.path.join(filename, '')
        tmp_import_path = []
        while os.path.basename(os.path.dirname(filename)) is not "":
            tmp_import_path.append(os.path.basename(os.path.dirname(filename)))
            filename = os.path.dirname(filename)
        importpath = ".".join(tmp_import_path[::-1])
        try:
            eq_dict = importlib.import_module(importpath)
            neuron_eq = eq_dict.__dict__[dict_name]
        except ImportError:
            spec = importlib.util.spec_from_file_location(
                dict_name[:-3], fallback_import_path)
            eq_dict = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eq_dict)
            neuron_eq = eq_dict.__dict__[dict_name[:-3]]

        builder_obj = cls(keywords=neuron_eq)
        builder_obj.add_input_currents(num_inputs)

        return builder_obj


def print_param_dictionaries(Dict):
    """Function to print dictionaries of parameters in an ordered way.

    Args:
        Dict (dict): Parameter dictionary to be printed.
    """
    for keys, values in Dict.items():
        print('      ' + keys + ' = ' + repr(values))


def print_neuron_model(Neuron_group):
    """Function to print keywords of a Neuron model
    Usefull to check the entire equation and parameter list

    Args:
       Neuron group( Neurons ) : Synaptic group

    NOTE: Even if mismatch is added, the values that are shown and not subject
    to mismatch
    """
    print("Neuron group: {}" .format(Neuron_group.equation_builder.keywords))
    return None
