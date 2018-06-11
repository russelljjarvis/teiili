# @Author: mrax, alpren, mmilde
# @Date:   2018-01-15 17:53:31
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2018-06-01 16:00:55


"""
This file contains a class that manages a synapse equation.

And it prepares a dictionary of keywords for easy synapse creation.

It also provides a function to add lines to the model.

"""
import os
import importlib
from brian2 import pF, nS, mV, ms, pA, nA
from teili.models.builder.combine import combine_syn_dict
from teili.models.builder.templates.synapse_templates import modes, kernels, plasticity_models,\
    current_parameters, conductance_parameters, DPI_parameters, DPI_shunt_parameters
import copy

class SynapseEquationBuilder():

    """Class which builds synapse equation.

    Attributes:
        keywords (dict): Total synapse model Brian2 compatible, composed of
            model (string): Actually neuron model differential equation
            on_post (string): Dictionary with equations specifying behaviour of synapse
                in response to a post-synaptic spike
            on_pre (string): Dictionary with equations specifying behaviour of synapse
                in response to a pre-synaptic spike
            parameters (dict): Dictionary of parameters
        verbose (bool): Flag to print more detailed output of neuron equation builder
    """

    def __init__(self, keywords=None, base_unit='current', kernel='exponential',
                 plasticity='non_plastic', verbose=False):
        """Summary

        Args:
            model (dict, optional): Brian2 model composed of model eq, on-pre eq,
                on-post eq, parameter dicionary
            base_unit (str, optional): Indicates if synapse is current-, conductance-based
                or a DPI current model (for reference see ) #let's add a paper here
            kernel (str, optional): Specifying temporal kernel with which each spike gets convolved, i.e.
                exponential, decay, or alpha function
            plasticity (str, optional): Plasticity algorithm for the synaptic weight. Can either be
                'non_plastic', 'fusi' or 'stdp'
            verbose (bool, optional): Flag to print more detailed output of neuron equation builder
        """
        self.verbose = verbose
        if keywords is not None:
            self.keywords = {'model': keywords['model'],
                             'on_pre': keywords['on_pre'],
                             'on_post': keywords['on_post'],
                             'parameters': keywords['parameters']}

        else:
            ERRValue = """
                                    ---Model not present in dictionaries---
                    This class constructor builds a model for a synapse using pre-existing blocks.

                    The first entry is the type of model;
                    choose between : 'current', 'conductance' or 'DPI' see this paper
                    for reference #(add DPI paper here)

                    The second entry is the kernel of the synapse.
                    This can be one of 'exponential', 'alpha', 'resonant' or 'gaussian'

                    The third entry is the plasticity of the synapse.
                    This can be 'non_plastic', 'stdp' or 'fusi' see this paper
                    for reference #(add fusi paper here)

                    """

            try:
                modes[base_unit]
                kernels[kernel]
                plasticity_models[plasticity]

            except KeyError:
                print(ERRValue)

            if base_unit == 'current':
                eq_tmpl = [modes[base_unit],
                           kernels[kernel],
                           plasticity_models[plasticity]]

                param_templ = [current_parameters[base_unit],
                               current_parameters[kernel],
                               current_parameters[plasticity]]

                keywords = combine_syn_dict(eq_tmpl, param_templ)

                keywords['model'] = keywords['model'].format(
                    input_number="{input_number}", unit='amp')
                keywords['on_pre'] = keywords['on_pre'].format(
                    input_number="{input_number}", unit='amp')
                keywords['on_post'] = keywords['on_post'].format(
                    input_number="{input_number}", unit='amp')

            if base_unit == 'conductance':
                eq_tmpl = [modes[base_unit],
                           kernels[kernel],
                           plasticity_models[plasticity]]

                param_templ = [conductance_parameters[base_unit],
                               conductance_parameters[kernel],
                               conductance_parameters[plasticity]]

                keywords = combine_syn_dict(eq_tmpl, param_templ)

                keywords['model'] = keywords['model'].format(
                    input_number="{input_number}", unit='siemens')
                keywords['on_pre'] = keywords['on_pre'].format(
                    input_number="{input_number}", unit='siemens')
                keywords['on_post'] = keywords['on_post'].format(
                    input_number="{input_number}", unit='siemens')

            if base_unit == 'DPI':
                eq_tmpl = [modes[base_unit],
                           kernels[kernel],
                           plasticity_models[plasticity]]

                param_templ = [DPI_parameters[base_unit],
                               DPI_parameters[kernel],
                               DPI_parameters[plasticity]]

                keywords = combine_syn_dict(eq_tmpl, param_templ)

                keywords['model'] = keywords['model'].format(
                    input_number="{input_number}", unit='amp')
                keywords['on_pre'] = keywords['on_pre'].format(
                    input_number="{input_number}", unit='amp')
                keywords['on_post'] = keywords['on_post'].format(
                    input_number="{input_number}", unit='amp')

            if base_unit == 'DPIShunting':
                eq_tmpl = [modes[base_unit],
                           kernels[kernel],
                           plasticity_models[plasticity]]

                param_templ = [DPI_shunt_parameters[base_unit],
                               DPI_shunt_parameters[kernel],
                               DPI_shunt_parameters[plasticity]]

                keywords = combine_syn_dict(eq_tmpl, param_templ)

                keywords['model'] = keywords['model'].format(
                    input_number="{input_number}", unit='amp')
                keywords['on_pre'] = keywords['on_pre'].format(
                    input_number="{input_number}", unit='amp')
                keywords['on_post'] = keywords['on_post'].format(
                    input_number="{input_number}", unit='amp')

            self.keywords = {'model': keywords['model'],
                             'on_pre': keywords['on_pre'],
                             'on_post': keywords['on_post'],
                             'parameters': keywords['parameters']}

        self.keywords_original = dict(self.keywords)

        if self.verbose == True:
            self.print_all()

    def __call__(self):
        """
        This allows the user to call the object like a class in order to make new objects.
        Maybe this use is a bit confusing, so rather not use it.
        """
        builder_copy =  copy.deepcopy(self)
        builder_copy.keywords = dict(self.keywords_original)
        return builder_copy


    def set_input_number(self, input_number):
        """Sets the input number of synapse. This is needed to overcome
        the summed issue in brian2.

        Args:
            input_number (int): Synapse's input number
        """
        self.keywords_original = self.keywords

        self.keywords['model'] = self.keywords['model'].format(
            input_number=str(input_number))  # input_number-1 ???
        self.keywords['on_pre'] = self.keywords[
            'on_pre'].format(input_number=str(input_number))
        self.keywords['on_post'] = self.keywords[
            'on_post'].format(input_number=str(input_number))

    def print_all(self):
        """Method to print all dictionaries within a synapse model
        """
        print('Model equation:')
        print(self.keywords['model'])
        print('-_-_-_-_-_-_-_-')
        print('Pre spike equation:')
        print(self.keywords['on_pre'])
        print('-_-_-_-_-_-_-_-')
        print('Post spike equation:')
        print(self.keywords['on_post'])
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
            file.write("'on_pre':\n")
            file.write("'''\n")
            file.write(self.keywords['on_pre'])
            file.write("''',\n")
            file.write("'on_post':\n")
            file.write("'''\n")
            file.write(self.keywords['on_post'])
            file.write("\n")
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
    def import_eq(cls, filename):
        #if only the filename without path is given, we assume it is one of the predefined models
        if os.path.dirname(filename) is "":
            filename = os.path.join('teili','models','equations',filename)

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
        synapse_eq = eq_dict.__dict__[dict_name]

        return cls(keywords=synapse_eq)


def print_param_dictionaries(Dict):
    """Function to print dictionaries of parameters in an ordered way

    Args:
        Dict (dict): Parameter dictionary to be printed
    """
    for keys, values in Dict.items():
        print('      ' + keys + ' = ' + repr(values))
