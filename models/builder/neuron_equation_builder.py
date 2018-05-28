# -*- coding: utf-8 -*-
# @Author: mrax, alpren, mmilde
# @Date:   2018-01-12 11:34:34
# @Last Modified by:   mrax
# @Last Modified time: 2018-04-16 00:03:10

"""
This file contains a class that manages a neuon equation

And it prepares a dictionary of keywords for easy synapse creation

It also provides a function to add lines to the model

"""
import os
from brian2 import pF, nS, mV, ms, pA, nA
from NCSBrian2Lib.models.builder.combine import combine_neu_dict
from NCSBrian2Lib.models.builder.templates.neuron_templates  import modes, currentEquationsets, voltageEquationsets, currentParameters, voltageParameters


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

    def __init__(self, model=None, baseUnit='current', adaptation='calciumFeedback',
                 integrationMode='exponential', leak='leaky', position='spatial',
                 noise='gaussianNoise', refractory = 'refractory', verbose=False):
        """Summary

        Args:
            model (dict, optional): Brian2 like model
            baseUnit (str, optional): Indicates if neuron is current- or conductance-based
            adaptation (str, optional): What type of adaptive feedback should be used.
               So far only calciumFeedback is implemented
            integrationMode (str, optional): Sets if integration up to spike-generation is
               rather linear or exponential
            leak (str, optional): Enables leaky integration
            position (str, optional): To enable spatial-like position indices to neuron
            noise (str, optional): NOT YET IMPLMENTED! This will in the future allow to
               add independent mismatch-like noise on each neuron.
            refractory (str, optional): Refractory period of the neuron
            verbose (bool, optional): Flag to print more detailed output of neuron equation builder
        """
        self.verbose = verbose
        if model is not None:
            keywords = model

        else:
            ERRValue = """
                                ---Model not present in dictionaries---
                    This class constructor build a model for a neuron using pre-existent blocks.

                    The first entry is the model type,
                    choice between : 'current' or 'voltage'

                    you can choose then what module load for you neuron,
                    the entries are 'adaptation', 'exponential', 'leaky', 'spatial', 'gaussianNoise'
                    if you don't want to load a module just use the keyword 'none'
                    example: NeuronEquationBuilder('current','none','expnential','leak','none','none'.....)

                    """

            try:
                modes[baseUnit]
                currentEquationsets[adaptation]
                currentEquationsets[integrationMode]
                currentEquationsets[leak]
                currentEquationsets[position]
                currentEquationsets[noise]

            except KeyError as e:
                print(ERRValue)

            if baseUnit == 'current':
                eq_templ = [modes[baseUnit],
                           currentEquationsets[adaptation],
                           currentEquationsets[integrationMode],
                           currentEquationsets[leak],
                           currentEquationsets[position],
                           currentEquationsets[noise],]
                param_templ = [currentParameters[baseUnit],
                               currentParameters[adaptation],
                               currentParameters[integrationMode],
                               currentParameters[leak],
                               currentParameters[position],
                               currentParameters[noise],]

                keywords = combine_neu_dict(eq_templ,param_templ)

            if baseUnit == 'voltage':
                eq_templ = [modes[baseUnit],
                           voltageEquationsets[adaptation],
                           voltageEquationsets[integrationMode],
                           voltageEquationsets[leak],
                           voltageEquationsets[position],
                           voltageEquationsets[noise],]
                param_templ = [voltageParameters[baseUnit],
                               voltageParameters[adaptation],
                               voltageParameters[integrationMode],
                               voltageParameters[leak],
                               voltageParameters[position],
                               voltageParameters[noise],]
                keywords = combine_neu_dict(eq_templ,param_templ)

            self.keywords = {'model': keywords['model'], 'threshold': keywords['threshold'],
                             'reset': keywords['reset'], 'refractory' : 'refP',  'parameters': keywords['parameters']}
            self.printAll()



    def printAll(self):
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
            file.write("'threshold':\n")
            file.write("'''\n")
            file.write(self.keywords['threshold'])
            file.write("\n")
            file.write("''',\n")
            file.write("'reset':\n")
            file.write("'''\n")
            file.write(self.keywords['reset'])
            file.write("\n")
            file.write("''',\n")
            file.write("'parameters':\n")
            file.write("{\n")
            for keys, values in self.keywords['parameters'].items():
                writestr = "'"+keys+"'"+' : '+repr(values)
                if 'famp' in writestr:
                    writestr = writestr.replace('famp','10**(-3) * pamp')
                file.write(writestr)
                file.write(",\n")
            file.write("}\n")
            file.write("}")

    def importeq(self, filename):
        print(filename)



def printParamDictionaries(Dict):
    """Function to print dictionaries of parameters in a ordered way

    Args:
        Dict (dict): Parameter dictionary to be printed
    """
    for keys, values in Dict.items():
        print('      '+keys+' = '+repr(values))

