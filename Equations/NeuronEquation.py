# coding: utf-8
"""
This file contains a class that manages a neuon equation

It automatically adds the line: Iin = Ie0 + Ii0 + Ie1 + Ii1 ...
And it prepares a dictionary of keywords for easy neurongroup creation

It also provides a funtion to add lines to the model

"""
from brian2 import *
from NCSBrian2Lib.Tools.tools import *
from brian2 import pF, nS, mV, ms, pA, nA
import operator


def combineEquations(*args):
    model = ''
    threshold = ''
    reset = ''
    varSet = {}
    varSet = set()
    for eq in args:
        if '%' in eq['model']:
            model, tmp = deleteVar(model, eq['model'], '%')
            varSet = set.union(tmp, varSet)
        else:
            model += eq['model']

        if '%' in eq['threshold']:
            threshold, tmp = deleteVar(threshold, eq['threshold'], '%')
            varSet = set.union(tmp, varSet)
        else:
            threshold += eq['threshold']

        if '%' in eq['reset']:
            reset, tmp = deleteVar(reset, eq['reset'], '%')
            varSet = set.union(tmp, varSet)

        else:
            reset += eq['reset']

    return {'model': model, 'threshold': threshold, 'reset': reset}, varSet


def deleteVar(firstEq, secondEq, var):
    varSet = {}
    varSet = set()
    resultfirstEq = ''
    resultsecondEq = ''
    for line in secondEq.splitlines():
        if var in line:  # for array variables
            var2 = line.split('%', 1)[1].split()[0]
            line = line.replace("%", "")
            if '/' in var2:
                var2 = var2.split('/', 1)[0][1:]
            diffvar2 = 'd' + var2 + '/dt'
            for line2 in firstEq.splitlines():

                if (var2 in line2) or (diffvar2 in line2):  # if i found a variable i need to check then if it's the explicit form we want to remove

                    if (var2 == line2.replace(':', '=').split('=', 1)[0].split()[0]) or (diffvar2 in line2.replace(':', '=').split('=', 1)[0].split()[0]):
                        varSet.add(var2)
                        pass
                    else:
                        resultfirstEq += line2 + "\n"

                else:
                    resultfirstEq += line2 + "\n"
            if len(line.split()) > 1:
                resultsecondEq += line + "\n"
            firstEq = resultfirstEq
            resultfirstEq = ""
        else:
            resultsecondEq += line + "\n"
    resultEq = firstEq + resultsecondEq
#    print(resultEq)
    return resultEq, varSet


def combineParDictionaries(varSet, *args):
    ParametersDict = {}
    for tmpDict in args:
        # OverrideList = list(ParametersDict.keys() & tmpDict.keys())
        # OverrideList = list(set(ParametersDict.keys()) & set(tmpDict.keys()))
        # OverrideList = list(ParametersDict.keys().intersection(tmpDict.keys()))
        OverrideList = list(set(ParametersDict.keys()).intersection(tmpDict.keys()))
        for key in OverrideList:
            ParametersDict.pop(key)
        ParametersDict.update(tmpDict)
    for key in list(varSet):
        if key in ParametersDict:
            ParametersDict.pop(key)
    return ParametersDict


class NeuronEquation():

    def __init__(self, model=None, baseUnit='current', adaptation='calciumFeedback', integrationMode='exponential', leak='leaky', position='spatial', noise='gaussianNoise', refractory=False, numInputs=1, additionalStatevars=None):
        if model is not None:
            eqDict = model
            self.model = eqDict['model']
            self.threshold = eqDict['threshold']
            self.reset = eqDict['reset']
            self.refractory = 'refP'
            self.parameters = eqDict['parameters']

            self.changeableParameters = ['refP']

            self.standaloneVars = {}  # TODO: this is just a dummy, needs to be written

            if additionalStatevars is not None:
                self.addStateVars(additionalStatevars)

            self.addInputCurrents(numInputs)

            self.keywords = {'model': self.model, 'threshold': self.threshold,
                             'reset': self.reset, 'refractory': self.refractory}
        else:
            ERRValue = """
                                ---Model not present in dictionaries---
                    This class constructor build a model for a neuron using pre-existent blocks.

                    The first entry is the model type,
                    choice between : 'current' or 'voltage'

                    you can choose then what module load for you neuron,
                    the entries are 'adaptation', 'exponential', 'leaky', 'spatial', 'gaussianNoise'
                    if you don't want to load a module just use the keyword 'none'
                    example: NeuronEquation('current','none','expnential','leak','none','none'.....)

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
                eqDict, varSet = combineEquations(modes[baseUnit], currentEquationsets[adaptation], i_a, currentEquationsets[integrationMode],
                                                  currentEquationsets[leak], currentEquationsets[position], currentEquationsets[noise])
                paraDict = combineParDictionaries(varSet, currentParameters[baseUnit], currentParameters[adaptation],
                                                  currentParameters[integrationMode], currentParameters[leak], currentParameters[noise])

            if baseUnit == 'voltage':
                eqDict, varSet = combineEquations(modes[baseUnit], voltageEquationsets[adaptation], voltageEquationsets[integrationMode],
                                                  voltageEquationsets[leak], voltageEquationsets[position], voltageEquationsets[noise])
                paraDict = combineParDictionaries(varSet, voltageParameters[baseUnit], voltageParameters[adaptation],
                                                  voltageParameters[integrationMode], voltageParameters[leak], voltageParameters[position], voltageParameters[noise])
                # paraDict = {}  # until we have parameters for voltage based equations

            if refractory is not False:
                refDict = {"refP": refractory}
                paraDict = combineParDictionaries([], paraDict, refDict)
            else:
                print("""
                      Refractory period has been set using a default value,
                      if you don't want a refractory period in your model just specify 'refractory=0*ms' in the constructor
                      """)

            self.model = eqDict['model']
            self.threshold = eqDict['threshold']
            self.reset = eqDict['reset']
            self.refractory = 'refP'
            self.parameters = paraDict

            self.changeableParameters = ['refP']

            self.standaloneVars = {}  # TODO: this is just a dummy, needs to be written

            if additionalStatevars is not None:
                self.addStateVars(additionalStatevars)

            self.addInputCurrents(numInputs)

            self.keywords = {'model': self.model, 'threshold': self.threshold,
                             'reset': self.reset, 'refractory': self.refractory}

    def addInputCurrents(self, numInputs):
        """automatically adds the line: Iin = Ie0 + Ii0 + Ie1 + Ii1 + ... + IeN + IiN (with N = numInputs)
        it also adds all thise input currents as statevariables"""
        Ies = ["Ie0"] + ["+ Ie" + str(i + 1) + " " for i in range(numInputs - 1)]
        Iis = ["+Ii0"] + ["+ Ii" + str(i + 1) + " " for i in range(numInputs - 1)]
        self.model = self.model + "Iin = " + "".join(Ies) + "".join(Iis) + " : amp # input currents\n"
        Iesline = ["    Ie" + str(i) + " : amp" for i in range(numInputs)]
        Iisline = ["    Ii" + str(i) + " : amp" for i in range(numInputs)]
        self.addStateVars(Iesline)
        self.model += "\n"
        self.addStateVars(Iisline)
        self.model += "\n"

    def addStateVars(self, stateVars):
        "just adds a line to the model equation"
        print("added to Equation: \n" + "\n".join(stateVars))
        self.model += "\n            ".join(stateVars)

    def printAll(self):
        printEqDict(self.keywords, self.parameters)

    def Brian2Eq(self):
        tmp = {'model': self.model,
               'threshold': self.threshold,
               'reset': self.reset}
        return tmp


# voltage based equation building blocks
v_model_template = {'model': """
         dVm/dt  = (Ileak + Iexp + Iin + Iconst + Inoise - Iadapt)/Cm  : volt (unless refractory)
         Ileak   : amp                            # leak current
         Iexp    : amp                            # exponential current
         Iadapt  : amp                             # adaptation current
         Inoise  : amp                             # noise current
         Iconst  : amp                             # additional input current
         Cm                 : farad     (shared, constant)    # membrane capacitance
         refP               : second    (shared, constant)    # refractory period (It is still possible to set it to False)
         Vthr               : volt      (shared)
         Vres               : volt      (shared, constant)        # reset potential
         """,
                    'threshold': "Vm > Vthr ",
                    'reset': "Vm = Vres "}

v_model_templatePara = {"Cm": 281 * pF,
                        "refP": 2 * ms,
                        "Ileak": 0 * pA,
                        "Iexp": 0 * pA,
                        "Iadapt": 0 * pA,
                        "Inoise": 0 * pA,
                        "Iconst": 0 * pA,
                        "Vthr": -50.4 * mV,
                        "Vres": -70.6 * mV

                        }

#                        "gL": 35 * nS,
#                        "EL": -70.6 * mV,
#                        "DeltaT": 2 * mV,
#                        "tauwad": 144 * ms,
#                        "a": 4 * nS,
#                        "b": 0.0805 * nA,
#                        "Vm": -70.6 * mV,
#                        "Iconst": 0 * pA,
#                        "sigma": 0 * pA,
#                        "taue": 5 * ms,
#                        "taui": 6 * ms,
#                        "EIe": 60.0 * mV,
#                        "EIi": -90.0 * mV

# exponential current (see exponential I&F Model)
v_expCurrent = {'model': """
            #exponential
            %Iexp = gL*DeltaT*exp((Vm - VT)/DeltaT) : amp
            VT      : volt      (shared, constant)        #
            DeltaT  : volt      (shared, constant)        # slope factor

            %Vthr = (VT + 5 * DeltaT) : volt  (shared)
            """,
                'threshold': '',
                'reset': ''}
# leak
v_leak = {'model': """
          #leak
          %Ileak = -gL*(Vm - EL) : amp
          gL      : siemens   (shared, constant)        # leak conductance
          EL      : volt      (shared, constant)        # leak reversal potential
         """,
          'threshold': '',
          'reset': ''}

# adaptation
v_adapt = {'model': """
        #adapt
        %dIadapt/dt = -(gAdapt*(EL - Vm) + Iadapt)/tauIadapt : amp
        tauIadapt  : second    (shared, constant)        # adaptation time constant
        gAdapt     : siemens   (shared, constant)        # adaptation decay parameter
        wIadapt    : amp       (shared, constant)        # adaptation weight
        """,
           'threshold': '',
           'reset': 'Iadapt += wIadapt; '}
# noise
v_noise = {'model': """
        %Inoise = xi*Anoise*(second**0.5) : amp
        Anoise  : amp       (constant)
        """,
           'threshold': '',
           'reset': ''}


# independent equation building blocks

# spatial location
spatial = {'model': """
           x : 1         (constant)        # x location on 2d grid (only set it if you need it)
           y : 1         (constant)        # y location on 2d grid
           """,
           'threshold': '',
           'reset': ''}


# activity
activity = {'model': """
        dActivity/dt = -Activity/tauAct : 1
        tauAct : second (shared, constant)
        """,
            'threshold': '',
            'reset': 'Activity += 1;'}


# none
none = {'model': '',
        'threshold': '',
        'reset': ''}


# Brette, Gerstner 2005 Exponential adaptive IF model
# see: http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model
#ExpAdaptIF = combineEquations(v_model_template, v_expCurrent, v_leak, v_adapt)
#
# ExpAdaptIF['default'] = {"Cm": 281 * pF,
#                         "gL": 35 * nS,
#                         "EL": -70.6 * mV,
#                         "VT": -50.4 * mV,
#                         "DeltaT": 2 * mV,
#                         "tauwad": 144 * ms,
#                         "a": 4 * nS,
#                         "b": 0.0805 * nA,
#                         "Vr": -70.6 * mV,
#                         "Vm": -70.6 * mV,
#                         "Iconst": 0 * pA,
#                         "refP": 2 * ms}
#
#
# simple leaky IF model
#simpleIF = combineEquations(v_model_template, v_leak)
# simpleIF['default'] = {"gL": 4.3 * nS,
#                       "Vr": -55 * mV,
#                       "EL": -55 * mV,
#                       "Vthr": -40 * mV,
#                       "Cm": 0.135 * pF,
#                       "Vm": -55 * mV
#                       }
#


# current based equation building blocks
# Silicon Neuron as in Chicca et al. 2014
# Author: Moritz Milde
# Code partially adapted from Daniele Conti and Llewyn Salt
# Email: mmilde@ini.uzh.chs
i_model_template = {'model': '''
            dImem/dt = (((Ith_clip / Itau_clip) * (Iin_clip  + Ia_clip - Ishunt - Iahp_clip)) - Ith_clip - ((1 + ((Ishunt - Iahp_clip - Ia_clip) / Itau_clip)) * Imem)   ) / (tau * ((Ith_clip/(Imem + Io)) + 1)) : amp (unless refractory)

            Iahp      : amp
            Ia        : amp
            Ia_clip   : amp
            Iahp_clip : amp

            Itau_clip = Itau*(Imem>Io) + Io*(Imem<=Io)  : amp
            Ith_clip = Ith*(Imem>Io) + Io*(Imem<=Io)    : amp
            Iin_clip = clip(Iin+Iconst,Io, 1*amp)       : amp

            tau = (Cmem * Ut) / (kappa * Itau_clip) : second        # Membrane time constant
            kappa = (kn + kp) / 2 : 1

            Inoise  : amp                                    # Noise due to mismatch

            kn      : 1 (shared, constant)                   # subthreshold slope factor for nFETs
            kp      : 1 (shared, constant)                   # subthreshold slope factor for pFETs
            Ut      : volt (shared, constant)                # Thermal voltage
            Io      : amp (shared, constant)                 # Dark current
            Cmem    : farad (shared, constant)               # Membrane capacitance
            Ispkthr : amp (constant)                         # Spiking threshold
            Ireset  : amp (shared, constant)                 # Reset current
            refP    : second    (shared, constant)           # refractory period (It is still possible to set it to False)
            Ith     : amp (constant)                         # DPI threshold (low pass filter).
            Itau    : amp (constant)                         # Leakage current
            Iconst  : amp (constant)                         # Additional input current similar to constant current injection
            Ishunt  : amp (constant)                         # Shunting inhibitory current (directly affects soma)
         ''',
                    'threshold': "Imem > Ispkthr",
                    'reset': "Imem = Ireset"}

i_model_templatePara = {
    #--------------------------------------------------------
    # Default equations disabled
    #--------------------------------------------------------
    "Ia": 0.5 * pA,                                # Feedback current
    "Iahp": 0.5 * pA,                                # Adaptation current
    "Inoise": 0.5 * pA,                                # Noise due to mismatch
    #--------------------------------------------------------
    # VLSI process parameters
    #--------------------------------------------------------
    "kn": 0.75,
    "kp": 0.66,
    "Ut": 25 * mV,
    "Io": 0.5 * pA,
    #---------------------------------------------------------
    # Silicon neuron parameters
    #---------------------------------------------------------
    "Cmem": 0.5 * pF,
    #    "Cahp": 0.5 * pF,
    #---------------------------------------------------------
    # Positive feedback parameters
    #---------------------------------------------------------
    "Iagain": 1 * nA,
    "Iath": 20 * nA,
    "Ianorm": 1 * nA,
    #---------------------------------------------------------
    # Adaptative and Calcium parameters
    #---------------------------------------------------------
    "Ica": 0.5 * pA,
    "Itauahp": 0.5 * pA,
    "Ithahp": 0.5 * pA,
    #---------------------------------------------------------
    # Shunting inhibition
    #---------------------------------------------------------
    "Ishunt": 0. * pA,
    #---------------------------------------------------------
    # Neuron parameters
    #---------------------------------------------------------
    "Ispkthr": 0.15 * nA,  # Spike threshold of excitatory neurons
    "Ireset": 1 * pA,  # Reset Imem to Ireset after each spike
    "Ith": 0.5 * pA,
    "Itau": 4.3 * pA,
    "refP": 1 * ms,
    #---------------------------------------------------------
    # Noise parameters
    #---------------------------------------------------------
    "mu": 0.25 * pA,
    "sigma": 0.1 * pA
}


# noise
i_noise = {'model': """
          mu : amp
          sigma : amp
          b = sign(2 * rand() -1) : 1 (constant over dt)
          %Inoise = b * (sigma * randn() + mu) : amp (constant over dt)

         """,
           'threshold': '',
           'reset': ''}

i_noisePara = {"mu": 0.25 * pA,
               "sigma": 0.1 * pA}

# feedback
i_a = {'model': """
        %Ia = Iagain / (1 + exp(-(Imem - Iath) / Ianorm)) : amp  # postive feedback current
        %Ia_clip = Ia*(Imem>Io) + 2*Io*(Imem<=Io)    : amp
        Iagain : amp (constant)
        Iath : amp (constant)
        Ianorm : amp (constant)

         """,
       'threshold': '',
       'reset': ''}

i_aPara = {"Iagain": 20 * nA,
           "Iath": 20 * nA,
           "Ianorm": 1 * nA}

# adaptation
i_ahp = {'model': """
          %dIahp/dt = (Ithahp * Ica / Itauahp - Ithahp - Iahp) / (tauahp * (Ithahp / Iahp + 1)) : amp # adaptation current
          %Iahp_clip = Iahp*(Imem>Io) + Io*(Imem<=Io)  : amp
          tauahp = (Cahp * Ut) / (kappa * Itauahp) : second # time constant of adaptation
          Iahpmax = (Ica / Itauahp) * Ithahp : amp # Ratio of currents through diffpair and adaptation block
          Ithahp : amp (constant)
          Itauahp : amp (constant)
          Cahp : farad (constant)
          tauca : second (constant)
         """,
         'threshold': '',
         'reset': '''
                  Iahp += Iahpmax
                  '''}

i_ahpPara = {"tauca": 20 * ms,
             "Itauahp": 120 * pA,
             "Ithahp": 20 * pA,
             "Ica": 30 * pA,
             "Cahp": 0.5 * pF}


i_exponentialPara = {"Ith": 0.9 * pA,
                     "Iath": 80 * pA,
                     "Iagain": 20 * pA,
                     "Ianorm": 7 * pA,
                     "Itau": 13 * pA}

i_nonLeakyPara = {"Itau": 0.5 * pA}

nonePara = {}

modes = {'current': i_model_template, 'voltage': v_model_template}

currentEquationsets = {'calciumFeedback': i_ahp, 'exponential': none, 'leaky': none, 'non-leaky': none,
                       'spatial': spatial, 'gaussianNoise': i_noise, 'none': none, 'linear': none}

voltageEquationsets = {'calciumFeedback': v_adapt, 'exponential': v_expCurrent, 'leaky': v_leak, 'non-leaky': none,
                       'spatial': spatial, 'gaussianNoise': v_noise, 'none': none, 'linear': none}

currentParameters = {'current': i_model_templatePara, 'calciumFeedback': i_ahpPara,
                     'exponential': i_exponentialPara, 'leaky': nonePara, 'non-leaky': i_nonLeakyPara,
                     'gaussianNoise': i_noisePara, 'none': nonePara, 'linear': nonePara}

voltageParameters = {'voltage': v_model_templatePara, 'calciumFeedback': nonePara,
                     'exponential': nonePara, 'leaky': nonePara, 'non-leaky': nonePara,
                     'gaussianNoise': nonePara, 'none': nonePara, 'linear': nonePara}


# simple leaky IF model
#currentExpAdapIF = combineEquations(i_model_template, i_a, i_ahp, i_noise)
# currentExpAdapIF['default'] = {"Itau": 20 * pA,
#                               "Ith": 17 * pA,
#                               "Ithahp": 15 * pA,
#                               "Itauahp": 20 * pA,
#                               "Ica": 10 * pA,
#                               "mu": 0.75 * pA,
#                               "sigma": 0.2 * pA,
#                               "Imem": 1 * pA
#                       }

# if you have a new equation, just add it to equations
#equations = {'ExpAdaptIF': ExpAdaptIF, 'simpleIF': simpleIF, 'Silicon': Silicon}

def printDictionaries(Dict):
    for keys, values in Dict.items():
        print(keys)
        print(repr(values))


def printEqDict(eqDict, param):
    print('Model equation:')
    print(eqDict['model'])
    print('-_-_-_-_-_-_-_-')
    print('threshold equation:')
    print(eqDict['threshold'])
    print('-_-_-_-_-_-_-_-')
    print('reset equation:')
    print(eqDict['reset'])
    print('-_-_-_-_-_-_-_-')
    print('')
    printDictionaries(param)
    print('-_-_-_-_-_-_-_-')
