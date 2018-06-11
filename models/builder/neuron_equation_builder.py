# -*- coding: utf-8 -*-
# @Author: mrax, alpren, mmilde
# @Date:   2018-01-12 11:34:34
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2018-06-11 18:16:20

"""
This file contains a class that manages a neuron equation.

And it prepares a dictionary of keywords for easy synapse creation.

It also provides a function to add lines to the model.

"""
import os
import importlib
import re
import copy
from brian2 import pF, nS, mV, ms, pA, nA
from teili.models.builder.combine import combine_neu_dict
from teili.models.builder.templates.neuron_templates import modes, current_equation_sets, voltage_equation_sets, \
    current_parameters, voltage_parameters


class NeuronEquationBuilder():

    """Class which builds neuron equation according to pre-defined properties such
    as spike-frequency adaptation, leakage etc.

    Attributes:
        changeableParameters (list): List of changeable parameters during runtime
        model (dict): Actually neuron model differential equation
        parameters (dict): Dictionary of parameters
        refractory (str): Refractory period of the neuron
        reset (str): Reset level after spike
        standaloneVars (dict): Dictionary of standalone variables
        threshold (str): Neuron's spiking threshold
        verbose (bool): Flag to print more detailed output of neuron equation builder
    """

    def __init__(self, keywords=None, base_unit='current', adaptation='calciumFeedback',
                 integration_mode='exponential', leak='leaky', position='spatial',
                 noise='gaussianNoise', refractory='refractory', num_inputs=1, verbose=False):
        """Summary

        Args:
            model (dict, optional): Brian2 like model
            base_unit (str, optional): Indicates if neuron is current- or conductance-based
            adaptation (str, optional): What type of adaptive feedback should be used.
               So far only calciumFeedback is implemented
            integration_mode (str, optional): Sets if integration up to spike-generation is
               linear or exponential
            leak (str, optional): Enables leaky integration
            position (str, optional): To enable spatial-like position indices on neuron
            noise (str, optional): NOT YET IMPLMENTED! This will in the future allow to
               add independent mismatch-like noise on each neuron.
            refractory (str, optional): Refractory period of the neuron
            verbose (bool, optional): Flag to print more detailed output of neuron equation builder
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
                    the entries are 'adaptation', 'exponential', 'leaky', 'spatial', 'gaussianNoise'.
                    If you don't want to load a module just use the keyword 'none'
                    example: NeuronEquationBuilder('current','none','exponential','leak','none','none'.....)

                    """

            try:
                modes[base_unit]
                current_equation_sets[adaptation]
                current_equation_sets[integration_mode]
                current_equation_sets[leak]
                current_equation_sets[position]
                current_equation_sets[noise]

            except KeyError as e:
                print(ERRValue)

            if base_unit == 'current':
                eq_templ = [modes[base_unit],
                            current_equation_sets[adaptation],
                            current_equation_sets[integration_mode],
                            current_equation_sets[leak],
                            current_equation_sets[position],
                            current_equation_sets[noise]]
                param_templ = [current_parameters[base_unit],
                               current_parameters[adaptation],
                               current_parameters[integration_mode],
                               current_parameters[leak],
                               current_parameters[position],
                               current_parameters[noise]]

                keywords = combine_neu_dict(eq_templ, param_templ)

            if base_unit == 'voltage':
                eq_templ = [modes[base_unit],
                            voltage_equation_sets[adaptation],
                            voltage_equation_sets[integration_mode],
                            voltage_equation_sets[leak],
                            voltage_equation_sets[position],
                            voltage_equation_sets[noise]]
                param_templ = [voltage_parameters[base_unit],
                               voltage_parameters[adaptation],
                               voltage_parameters[integration_mode],
                               voltage_parameters[leak],
                               voltage_parameters[position],
                               voltage_parameters[noise]]
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
        """
        This allows the user to call the object with the num_inputs argument, like it is done with the class
        Maybe this use is a bit confusing, but it may be convenient.
        """
        builder_copy = copy.deepcopy(self)
        builder_copy.add_input_currents(num_inputs)
        return builder_copy

    def add_input_currents(self, num_inputs):
        """automatically adds the line: Iin = Ie0 + Ii0 + Ie1 + Ii1 + ... + IeN + IiN (with N = num_inputs)
        it also adds all these input currents as state variables

        Args:
            num_inputs (int): Number of inputs to the post-synaptic neuron
        """
        self.num_inputs = num_inputs
        # remove previously added inputcurrent lines
        inputcurrent_e_pattern = re.compile("Ie\d+ : amp")
        inputcurrent_i_pattern = re.compile("Ii\d+ : amp")
        model = self.keywords['model'].split('\n')
        for line in self.keywords['model'].split('\n'):
            if "Iin =" in line or "Iin=" in line:
                model.remove(line)
                if self.verbose:
                    print(
                        'previously added input currents were removed, following lines deleted:')
                    print(line)
            elif inputcurrent_e_pattern.search(line) is not None:
                if self.verbose:
                    print(line)
                model.remove(line)
            elif inputcurrent_i_pattern.search(line) is not None:
                model.remove(line)
                if self.verbose:
                    print(line)

        self.keywords['model'] = '\n'.join(model)

        Ies = ["Ie0"] + ["+ Ie" +
                         str(i + 1) + " " for i in range(num_inputs - 1)]
        Iis = ["+Ii0"] + ["+ Ii" +
                          str(i + 1) + " " for i in range(num_inputs - 1)]

        self.keywords['model'] = self.keywords['model'] + "\nIin = " + \
            "".join(Ies) + "".join(Iis) + " : amp # input currents\n"
        Iesline = ["        Ie" + str(i) + " : amp" for i in range(num_inputs)]
        Iisline = ["        Ii" + str(i) + " : amp" for i in range(num_inputs)]
        self.add_state_vars(Iesline)
        self.keywords['model'] += "\n"
        self.add_state_vars(Iisline)
        self.keywords['model'] += "\n"

    def add_state_vars(self, stateVars):
        """this function adds state variables to neuron equation by just adding
        a line to the neuron model equation.

        Args:
            stateVars (dict): State variable to be added to neuron model
        """
        if self.verbose:
            print("added to Equation: \n" + "\n".join(stateVars))
        self.keywords['model'] += "\n            ".join(stateVars)

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
        '''
        num_inputs is used to add additional input currents that are used
        for different synapses that are summed
        '''
        # if only the filename without path is given, we assume it is one of
        # the predefined models
        if os.path.dirname(filename) is "":
            filename = os.path.join('teili', 'models', 'equations', filename)

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

        eq_dict = importlib.import_module(importpath)
        neuron_eq = eq_dict.__dict__[dict_name]

        builder_obj = cls(keywords=neuron_eq)
        builder_obj.add_input_currents(num_inputs)

        return builder_obj


def print_param_dictionaries(Dict):
    """Function to print dictionaries of parameters in an ordered way

    Args:
        Dict (dict): Parameter dictionary to be printed
    """
    for keys, values in Dict.items():
        print('      ' + keys + ' = ' + repr(values))
