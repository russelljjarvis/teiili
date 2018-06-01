# @Author: mrax, alpren, mmilde
# @Date:   2018-01-15 17:53:31
# @Last Modified by:   mmilde
# @Last Modified time: 2018-05-29 12:24:16


"""
This file contains a class that manages a synapse equation

And it prepares a dictionary of keywords for easy synapse creation

It also provides a function to add lines to the model

"""
import os
import importlib
from brian2 import pF, nS, mV, ms, pA, nA
from NCSBrian2Lib.models.builder.combine import combine_syn_dict
from NCSBrian2Lib.models.builder.templates.synapse_templates import modes, kernels, plasticitymodels, current_Parameters, conductance_Parameters, DPI_Parameters


class SynapseEquationBuilder():

    """Class which builds synapse equation

    Attributes:
        keywords (dict): Total synapse model Brian2 compatible, composed by
            model (string): Actually neuron model differential equation
            on_post (string): Dictionary with equations specifying behaviour of synapse to
                post-synaptic spike
            on_pre (string): Dictionary with equations specifying behaviour of synapse to
                pre-synaptic spike
            parameters (dict): Dictionary of parameters
        verbose (bool): Flag to print more detailed output of neuron equation builder
    """

    def __init__(self, keywords=None, baseUnit='current', kernel='exponential',
                 plasticity='nonplastic', verbose=False):
        """Summary

        Args:
            model (dict, optional): Brian2 model composed by model eq, on-pre eq,
                on-post eq, parameter dicionary
            baseUnit (str, optional): Indicates if synapse is current-, conductance-based
                or a DPI current model (for reference see ) #let's add a paper here
            kernel (str, optional): Specifying temporal kernel with which each spike gets convolved, i.e.
                exponential, decay, or alpha function
            plasticity (str, optional): Plasticity algorithm for the synaptic weight. Can either be
                'nonplastic', 'fusi' or 'stdp'
            verbose (bool, optional): Flag to print more detailed output of neuron equation builder
        """
        self.verbose = verbose
        if keywords is not None:
            self.keywords = {'model': keywords['model'], 'on_pre': keywords['on_pre'],
                             'on_post': keywords['on_post'], 'parameters': keywords['parameters']}

        else:
            ERRValue = """
                                    ---Model not present in dictionaries---
                    This class constructor build a model for a synapse using pre-existent blocks.

                    The first entry is the type of model,
                    choice between : 'current', 'conductance' or 'DPI' see this paper
                    for reference #(add DPI paper here)

                    The second entry is the kernel of the synapse
                    can be one of those : 'exponential', 'alpha', 'resonant' or 'gaussian'

                    The third entry is the plasticity of the synapse
                    can be : 'nonplastic', 'stdp' or 'fusi' see this paper
                    for reference #(add fusi paper here)

                    """

            try:
                modes[baseUnit]
                kernels[kernel]
                plasticitymodels[plasticity]

            except KeyError:
                print(ERRValue)

            if baseUnit == 'current':
                eq_tmpl = [modes[baseUnit],
                          kernels[kernel],
                          plasticitymodels[plasticity]]

                param_templ = [current_Parameters[baseUnit],
                               current_Parameters[kernel],
                               current_Parameters[plasticity]]

                keywords = combine_syn_dict(eq_tmpl, param_templ)

                keywords['model'] = keywords['model'].format(inputnumber="{inputnumber}", unit='amp')
                keywords['on_pre'] = keywords['on_pre'].format(inputnumber="{inputnumber}", unit='amp')
                keywords['on_post'] = keywords['on_post'].format(inputnumber="{inputnumber}", unit='amp')

            if baseUnit == 'conductance':
                eq_tmpl = [modes[baseUnit],
                           kernels[kernel],
                           plasticitymodels[plasticity]]

                param_templ = [conductance_Parameters[baseUnit],
                               conductance_Parameters[kernel],
                               conductance_Parameters[plasticity]]

                keywords = combine_syn_dict(eq_tmpl, param_templ)

                keywords['model'] = keywords['model'].format(inputnumber="{inputnumber}", unit='siemens')
                keywords['on_pre'] = keywords['on_pre'].format(inputnumber="{inputnumber}", unit='siemens')
                keywords['on_post'] = keywords['on_post'].format(inputnumber="{inputnumber}", unit='siemens')

            if baseUnit == 'DPI':
                eq_tmpl = [modes[baseUnit],
                           kernels[kernel],
                           plasticitymodels[plasticity]]

                param_templ = [DPI_Parameters[baseUnit],
                               DPI_Parameters[kernel],
                               DPI_Parameters[plasticity]]

                keywords = combine_syn_dict(eq_tmpl, param_templ)

                keywords['model'] = keywords['model'].format(inputnumber="{inputnumber}", unit='amp')
                keywords['on_pre'] = keywords['on_pre'].format(inputnumber="{inputnumber}", unit='amp')
                keywords['on_post'] = keywords['on_post'].format(inputnumber="{inputnumber}", unit='amp')

            self.keywords = {'model': keywords['model'], 'on_pre': keywords['on_pre'],
                             'on_post': keywords['on_post'], 'parameters': keywords['parameters']}

        if self.verbose == True:
            self.printAll()

    def set_inputnumber(self, inputnumber):
        """Sets the respective input number of synapse. This is needed to overcome
        the summed issue in brian2.

        Args:
            inputnumber (int): Synapse's input number
        """
        self.keywords['model'] = self.keywords['model'].format(inputnumber=str(inputnumber - 1))  # inputnumber-1 ???
        self.keywords['on_pre'] = self.keywords['on_pre'].format(inputnumber=str(inputnumber - 1))
        self.keywords['on_post'] = self.keywords['on_post'].format(inputnumber=str(inputnumber - 1))

    def printAll(self):
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
        printParamDictionaries(self.keywords['parameters'])
        print('-_-_-_-_-_-_-_-')

    def exporteq(self, filename):
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
    def importeq(cls, filename):
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


def printParamDictionaries(Dict):
    """Function to print dictionaries of parameters in a ordered way

    Args:
        Dict (dict): Parameter dictionary to be printed
    """
    for keys, values in Dict.items():
        print('      ' + keys + ' = ' + repr(values))
