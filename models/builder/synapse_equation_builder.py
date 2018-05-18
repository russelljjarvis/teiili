# @Author: mrax, alpren, mmilde
# @Date:   2018-01-15 17:53:31
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-25 15:37:52


"""
This file contains a class that manages a synapse equation

And it prepares a dictionary of keywords for easy synapse creation

It also provides a function to add lines to the model

"""
import json
from brian2 import pF, nS, mV, ms, pA, nA
from NCSBrian2Lib.models.builder.combine import combine_syn_dict
from NCSBrian2Lib.models.builder.templates.synapse_templates  import modes, kernels, plasticitymodels, current_Parameters, conductance_Parameters, DPI_Parameters

 

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

    def __init__(self, model=None, baseUnit='current', kernel='exponential',
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
        if model is not None:
            keywords = model

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

            except KeyError as e:
                print(ERRValue)
                

            if baseUnit == 'current':
                keywords = combine_syn_dict(modes[baseUnit], kernels[kernel],
                                            plasticitymodels[plasticity],
                                            current_Parameters[baseUnit],
                                            current_Parameters[kernel],
                                            current_Parameters[plasticity])
                keywords['model'] = keywords['model'].format(unit='amp')
                keywords['on_pre'] = keywords['on_pre'].format(unit='amp')
                keywords['on_post'] = keywords['on_post'].format(unit='amp')



            if baseUnit == 'conductance':
                keywords = combine_syn_dict(modes[baseUnit], kernels[kernel],
                                            plasticitymodels[plasticity],
                                            conductance_Parameters[baseUnit],
                                            conductance_Parameters[kernel],
                                            conductance_Parameters[plasticity])
                keywords['model'] = keywords['model'].format(unit='siemens')
                keywords['on_pre'] = keywords['on_pre'].format(unit='siemens')
                keywords['on_post'] = keywords['on_post'].format(unit='siemens')



            if  baseUnit == 'DPI':
                keywords = combine_syn_dict(modes[baseUnit], kernels[kernel],
                                            plasticitymodels[plasticity],
                                            DPI_Parameters[baseUnit],
                                            DPI_Parameters[kernel],
                                            DPI_Parameters[plasticity])
                keywords['model'] = keywords['model'].format(unit='amp')
                keywords['on_pre'] = keywords['on_pre'].format(unit='amp')
                keywords['on_post'] = keywords['on_post'].format(unit='amp')




            self.keywords = {'model': keywords['model'], 'on_pre': keywords['on_pre'],
                             'on_post': keywords['on_post'], 'parameters': keywords['parameters']}
            
            if self.verbose == True:
                self.printAll()


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



    def exporteq(self, namefile):
        with open(namefile + ".txt", 'w') as file:
            json.dump(self.keywords, file)   
    
    def importeq(self, namefile):
        print(namefile)


def printParamDictionaries(Dict):
    """Function to print dictionaries of parameters in a ordered way

    Args:
        Dict (dict): Parameter dictionary to be printed
    """
    for keys, values in Dict.items():
        print('      '+keys+' = '+repr(values))

